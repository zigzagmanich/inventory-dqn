import sys
import os
import random
import numpy as np
import pandas as pd
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.dates as mdates
from datetime import datetime, timedelta

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QStackedWidget,
    QFrame, QGridLayout, QProgressBar, QSizePolicy,
    QSpinBox, QDoubleSpinBox, QMessageBox, QTabWidget, QTextEdit,
    QGroupBox, QScrollArea
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont


#  ЦВЕТА И СТИЛИ

DARK   = "#F0F4F8"
PANEL  = "#FFFFFF"
CARD   = "#FFFFFF"
BORDER = "#C8D4E0"
ACCENT = "#1A6FE0"
ACCENT2= "#D63030"
ACCENT3= "#0A7A5A"
TEXT   = "#1A1F2E"
MUTED  = "#5A6480"
WARN   = "#B86A00"

SS = f"""
QMainWindow, QWidget {{
    background-color: {DARK};
    color: {TEXT};
    font-family: Arial, Helvetica, sans-serif;
    font-size: 14px;
}}
QLabel {{ color: {TEXT}; background: transparent; font-size: 14px; }}
QPushButton {{
    background-color: {CARD};
    color: {ACCENT};
    border: 2px solid {ACCENT};
    border-radius: 6px;
    padding: 9px 20px;
    font-family: Arial, Helvetica, sans-serif;
    font-size: 13px;
    font-weight: bold;
}}
QPushButton:hover {{ background-color: {ACCENT}; color: #FFFFFF; }}
QPushButton:disabled {{ background-color: #E8ECF0; color: {MUTED}; border: 2px solid {BORDER}; }}
QPushButton#danger {{ color: {ACCENT2}; border: 2px solid {ACCENT2}; }}
QPushButton#danger:hover {{ background-color: {ACCENT2}; color: #FFFFFF; }}
QPushButton#primary {{ background-color: {ACCENT}; color: #FFFFFF; font-size: 14px; border: none; }}
QPushButton#primary:hover {{ background-color: #1558B8; }}
QPushButton#home {{ color: {MUTED}; border: 2px solid {BORDER}; font-size: 13px; padding: 7px 16px; background: {CARD}; }}
QPushButton#home:hover {{ color: {TEXT}; border: 2px solid {MUTED}; background: {DARK}; }}
QLineEdit, QSpinBox, QDoubleSpinBox {{
    background-color: {CARD};
    color: {TEXT};
    border: 2px solid {BORDER};
    border-radius: 6px;
    padding: 7px 12px;
    font-family: Arial, Helvetica, sans-serif;
    font-size: 14px;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border: 2px solid {ACCENT}; }}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ width: 20px; }}
QProgressBar {{
    background-color: #E4EAF0;
    border: 1px solid {BORDER};
    border-radius: 6px;
    text-align: center;
    color: {TEXT};
    font-size: 13px;
    font-family: Arial;
}}
QProgressBar::chunk {{ background-color: {ACCENT}; border-radius: 5px; }}
QScrollArea {{ border: none; background: transparent; }}
QTabWidget::pane {{ border: 1px solid {BORDER}; background: {PANEL}; }}
QTabBar::tab {{
    background: {DARK}; color: {MUTED};
    padding: 10px 24px; border: 1px solid {BORDER};
    font-family: Arial; font-size: 13px; font-weight: bold;
}}
QTabBar::tab:selected {{ background: {PANEL}; color: {ACCENT}; border-bottom: 3px solid {ACCENT}; }}
QTabBar::tab:hover {{ color: {TEXT}; }}
QGroupBox {{
    border: 2px solid {BORDER}; border-radius: 8px;
    margin-top: 14px; padding-top: 10px;
    font-family: Arial; color: {MUTED};
    font-size: 12px; font-weight: bold;
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 6px; background: {DARK}; }}
QTextEdit {{
    background-color: {CARD}; color: {TEXT};
    border: 2px solid {BORDER};
    font-family: 'Courier New', monospace; font-size: 13px;
}}
QScrollBar:vertical {{ background: {DARK}; width: 10px; border-radius: 5px; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 5px; min-height: 24px; }}
QScrollBar::handle:vertical:hover {{ background: {MUTED}; }}
"""

#  DQN ЯДРО

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

class DQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 64), nn.ReLU(),
            nn.Linear(64, 64),         nn.ReLU(),
            nn.Linear(64, action_size)
        )
    def forward(self, x): return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=50000): self.buffer = deque(maxlen=capacity)
    def push(self, s,a,r,s2,d): self.buffer.append((s,a,r,s2,d))
    def sample(self, n):
        b = random.sample(self.buffer, n)
        s,a,r,s2,d = zip(*b)
        return np.array(s),np.array(a),np.array(r),np.array(s2),np.array(d)
    def __len__(self): return len(self.buffer)

class DQNAgent:
    def __init__(self, state_size, action_size, lr=5e-4, gamma=0.99,
                 eps_start=1.0, eps_min=0.05, eps_decay=0.997):
        self.gamma=gamma; self.epsilon=eps_start
        self.eps_min=eps_min; self.eps_decay=eps_decay
        self.n_actions=action_size
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net=DQNetwork(state_size,action_size).to(self.device)
        self.target_net=DQNetwork(state_size,action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer=optim.Adam(self.policy_net.parameters(),lr=lr)
        self.memory=ReplayBuffer()
        self.batch_size=64; self.learn_every=4; self.step_count=0

    def select_action(self, state, explore=True):
        if explore and random.random()<self.epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            s=torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.policy_net(s).argmax().item()

    def train_step(self):
        self.step_count+=1
        if self.step_count%self.learn_every!=0 or len(self.memory)<self.batch_size: return
        s,a,r,s2,d=self.memory.sample(self.batch_size)
        s=torch.FloatTensor(s).to(self.device); a=torch.LongTensor(a).to(self.device)
        r=torch.FloatTensor(r).to(self.device); s2=torch.FloatTensor(s2).to(self.device)
        d=torch.FloatTensor(d).to(self.device)
        q=self.policy_net(s).gather(1,a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            na=self.policy_net(s2).argmax(1)
            nq=self.target_net(s2).gather(1,na.unsqueeze(1)).squeeze(1)
            target=r+(1-d)*self.gamma*nq
        loss=F.smooth_l1_loss(q,target)
        self.optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(),1.0)
        self.optimizer.step()

    def update_target(self): self.target_net.load_state_dict(self.policy_net.state_dict())
    def decay_epsilon(self): self.epsilon=max(self.eps_min, self.epsilon*self.eps_decay)


#  СРЕДА

class InventoryEnv:
    def __init__(self, df, params):
        self.df=df.reset_index(drop=True); self.p=params
        self.DEMAND=self.df['yesterday_demand'].to_numpy().copy()
        self.DAY_OF_YEAR=self.df['day_of_year'].to_numpy()
        self.demand_mean=max(1.0,self.DEMAND.mean())
        self.demand_std=self.DEMAND.std() if self.DEMAND.std()>0 else 1.0
        self.seasonal_mean=self.df.groupby('day_of_year')['yesterday_demand'].mean().to_dict()
        n=int(params['capacity']//3//50)+1
        self.ACTIONS=[i*50 for i in range(n)]; self.N_ACTIONS=len(self.ACTIONS)
        self.reset()

    def reset(self):
        self.day_idx=0; self.inv=self.p['init_inventory']
        self.demand_history=deque(maxlen=30); self.pending_orders=deque()
        self.demand_history.append(self.DEMAND[0])
        return self._get_state()

    def _get_state(self):
        state=[]
        state.append(self.inv/self.p['capacity'])
        d=self.DAY_OF_YEAR[self.day_idx]
        state.append(np.sin(2*np.pi*d/365)); state.append(np.cos(2*np.pi*d/365))
        hist=list(self.demand_history)
        sm=self.seasonal_mean.get(d,self.demand_mean)
        for w in [7,14,30]:
            state.append(np.mean(hist[-w:])/sm if len(hist)>=w else 1.0)
        state.append(hist[-1]/sm)
        if len(hist)>=14:
            state.append(np.clip((np.mean(hist[-7:])-np.mean(hist[-14:-7]))/sm,-2,2))
        else: state.append(0.0)
        for off in range(7):
            do=(d+off)%365
            state.append(self.seasonal_mean.get(do,self.demand_mean)/self.demand_mean)
        pq=sum(q for _,q in self.pending_orders); pc=len(self.pending_orders)
        state.append(pq/self.p['capacity'])
        state.append(np.clip(pc/self.p['lead_time'],0,1))
        return np.array(state,dtype=np.float32)

    def step(self, action_idx):
        action=self.ACTIONS[action_idx]
        delivered=0; new_q=deque()
        for dl,qty in self.pending_orders:
            if dl==0: self.inv=min(self.p['capacity'],self.inv+qty); delivered+=qty
            else: new_q.append((dl-1,qty))
        self.pending_orders=new_q
        if action>0: self.pending_orders.append((self.p['lead_time']-1,action))
        demand=self.DEMAND[self.day_idx]
        sold=min(self.inv,demand); shortage=max(0,demand-self.inv); self.inv-=sold
        rev=sold*self.p['selling_price']; hold=self.inv*self.p['holding_cost']
        proc=action*self.p['order_unit_cost']
        fix=self.p['order_fixed_cost'] if action>0 else 0.0
        stk=shortage*self.p['stockout_penalty']
        reward=(rev-hold-proc-fix-stk)/100.0+sold/50.0
        self.demand_history.append(demand); self.day_idx+=1
        done=self.day_idx>=len(self.DEMAND)
        ns=None if done else self._get_state()
        info={"sold":sold,"shortage":shortage,"order":action,"inventory":self.inv,
              "demand":demand,"delivered":delivered,"pending":list(self.pending_orders)}
        return ns,reward,done,info

def auto_penalty(params):
    return round((params['selling_price']+params['order_unit_cost'])*params['lead_time']*4,1)


#  ПОТОК ОБУЧЕНИЯ

class TrainWorker(QThread):
    progress    = pyqtSignal(int, float, float)
    log_line    = pyqtSignal(str)
    finished_ok = pyqtSignal(object, list, list)

    def __init__(self, df_train, params, episodes):
        super().__init__()
        self.df_train=df_train; self.params=params; self.episodes=episodes; self._stop=False

    def stop(self): self._stop=True

    def run(self):
        try:
            env=InventoryEnv(self.df_train,self.params)
            state=env.reset()
            agent=DQNAgent(len(state),env.N_ACTIONS)
            rewards,epsilons=[],[]
            for ep in range(1,self.episodes+1):
                if self._stop: return
                state=env.reset(); total_r=0
                while True:
                    a=agent.select_action(state)
                    ns,r,done,_=env.step(a)
                    agent.memory.push(state,a,r,np.zeros_like(state) if ns is None else ns,done)
                    agent.train_step(); total_r+=r
                    if done: break
                    state=ns
                agent.decay_epsilon()
                if ep%10==0: agent.update_target()
                rewards.append(total_r*100); epsilons.append(agent.epsilon)
                self.progress.emit(ep,total_r*100,agent.epsilon)
                if ep%50==0:
                    self.log_line.emit(f"  Эпизод {ep:4d}  |  награда {total_r*100:8.0f}  |  ε {agent.epsilon:.3f}")
            self.finished_ok.emit(agent,rewards,epsilons)
        except Exception:
            import traceback
            self.log_line.emit('  ✗ ОШИБКА В ПОТОКЕ ОБУЧЕНИЯ:\n' + traceback.format_exc())


#  CANVAS

class MplCanvas(FigureCanvas):
    def __init__(self, figsize=(7,3), nrows=1, ncols=1):
        self.fig,self.axes=plt.subplots(nrows,ncols,figsize=figsize,facecolor=CARD)
        super().__init__(self.fig)
        self.setStyleSheet(f"background-color:{CARD};")
        self.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
    def ax(self,i=0):
        return self.axes[i] if hasattr(self.axes,'__len__') else self.axes


#  ВСПОМОГАТЕЛЬНЫЕ ВИДЖЕТЫ

def home_btn(cb):
    b=QPushButton("⌂  НА ГЛАВНУЮ"); b.setObjectName("home"); b.setFixedHeight(32)
    b.clicked.connect(cb); return b

def page_header(title, subtitle=None, cb_home=None):
    w=QWidget(); lay=QHBoxLayout(w); lay.setContentsMargins(0,0,0,0)
    left=QVBoxLayout()
    t=QLabel(title); t.setFont(QFont("Courier New",20,QFont.Bold))
    t.setStyleSheet(f"color:{ACCENT}; letter-spacing:4px;"); left.addWidget(t)
    if subtitle:
        s=QLabel(subtitle); s.setStyleSheet(f"color:{MUTED}; font-size:13px;")
        left.addWidget(s)
    lay.addLayout(left); lay.addStretch()
    if cb_home: lay.addWidget(home_btn(cb_home))
    return w

def info_card(title, color=None):
    f=QFrame()
    f.setStyleSheet(f"background:{CARD}; border:1px solid {color or BORDER}; border-radius:8px; padding:12px;")
    vl=QVBoxLayout(f)
    t=QLabel(title); t.setStyleSheet(f"color:{MUTED}; font-size:13px;")
    vl.addWidget(t)
    v=QLabel("—"); v.setFont(QFont("Courier New",18,QFont.Bold))
    v.setStyleSheet(f"color:{color or TEXT};"); vl.addWidget(v)
    return f,v


#  СТРАНИЦА 1: НАСТРОЙКА

class SetupPage(QWidget):
    go_train=pyqtSignal(object,object,object)

    def __init__(self):
        super().__init__(); self.df=None; self._build()

    def _build(self):
        root=QVBoxLayout(self); root.setSpacing(14); root.setContentsMargins(40,28,40,28)
        root.addWidget(page_header("УПРАВЛЕНИЕ ЗАПАСАМИ","обучение с подкреплением · DQN агент"))
        root.addSpacing(4)

        # Файл
        fg=QGroupBox("ФАЙЛ ДАННЫХ  (Excel, лист State_Input)")
        fl=QHBoxLayout(fg)
        self.file_lbl=QLabel("Файл не выбран"); self.file_lbl.setStyleSheet(f"color:{MUTED};")
        btn=QPushButton("ОТКРЫТЬ"); btn.clicked.connect(self._load)
        fl.addWidget(self.file_lbl,1); fl.addWidget(btn)
        root.addWidget(fg)
        self.split_lbl=QLabel(""); self.split_lbl.setStyleSheet(f"color:{MUTED}; font-size:13px; padding-left:4px;")
        root.addWidget(self.split_lbl)

        # Параметры склада
        wg=QGroupBox("ПАРАМЕТРЫ СКЛАДА"); gl=QGridLayout(wg); gl.setSpacing(10)
        self.pw={}
        defs=[
            ("Начальный запас (ед.)",   "init_inventory",   500,  0,99999,False,50),
            ("Вместимость склада (ед.)","capacity",         1500,100,999999,False,50),
            ("Лаг поставки (дней)",     "lead_time",        3,    1,60,   False,1),
            ("Цена продажи",            "selling_price",    8.0,  0.1,9999,True,0.5),
            ("Цена закупки (за ед.)",   "order_unit_cost",  5.0,  0.1,9999,True,0.5),
            ("Фикс. стоимость заказа",  "order_fixed_cost", 40.0, 0,  9999,True,5.0),
            ("Стоимость хранения/ед.",  "holding_cost",     0.1,  0,  100, True,0.05),
        ]
        for i,(lbl,key,default,lo,hi,isfloat,step) in enumerate(defs):
            row,col=divmod(i,2)
            l=QLabel(lbl); l.setStyleSheet(f"color:{MUTED}; font-size:13px;")
            if isfloat:
                s=QDoubleSpinBox(); s.setDecimals(2); s.setSingleStep(step)
            else:
                s=QSpinBox(); s.setSingleStep(int(step))
            s.setRange(lo,hi); s.setValue(default)
            self.pw[key]=s
            gl.addWidget(l,row,col*2); gl.addWidget(s,row,col*2+1)
        root.addWidget(wg)

        # Штраф и эпохи
        pg=QGroupBox("ШТРАФ ЗА ДЕФИЦИТ И ОБУЧЕНИЕ"); pl=QGridLayout(pg); pl.setSpacing(10)
        l1=QLabel("Штраф за дефицит (stockout penalty)"); l1.setStyleSheet(f"color:{MUTED}; font-size:13px;")
        self.pw['stockout_penalty']=QDoubleSpinBox()
        self.pw['stockout_penalty'].setRange(0,999999); self.pw['stockout_penalty'].setValue(96.0)
        self.pw['stockout_penalty'].setSingleStep(10); self.pw['stockout_penalty'].setDecimals(1)
        self.auto_lbl=QLabel(""); self.auto_lbl.setStyleSheet(f"color:{ACCENT}; font-size:13px;")
        btn_auto=QPushButton("АВТО"); btn_auto.setFixedWidth(72); btn_auto.clicked.connect(self._set_auto)

        l2=QLabel("Количество эпох обучения"); l2.setStyleSheet(f"color:{MUTED}; font-size:13px;")
        self.pw['episodes']=QSpinBox(); self.pw['episodes'].setRange(50,10000)
        self.pw['episodes'].setValue(500); self.pw['episodes'].setSingleStep(50)

        pl.addWidget(l1,0,0); pl.addWidget(self.pw['stockout_penalty'],0,1)
        pl.addWidget(btn_auto,0,2); pl.addWidget(self.auto_lbl,0,3)
        pl.addWidget(l2,1,0); pl.addWidget(self.pw['episodes'],1,1)
        root.addWidget(pg)

        for k in ['selling_price','order_unit_cost','lead_time']:
            self.pw[k].valueChanged.connect(self._refresh_auto)
        self._refresh_auto()

        self.btn_start=QPushButton("▶  НАЧАТЬ ОБУЧЕНИЕ")
        self.btn_start.setObjectName("primary"); self.btn_start.setFixedHeight(46)
        self.btn_start.clicked.connect(self._start); self.btn_start.setEnabled(False)
        root.addWidget(self.btn_start); root.addStretch()

    def _refresh_auto(self):
        p={k:self.pw[k].value() for k in ['selling_price','order_unit_cost','lead_time']}
        v=auto_penalty(p)
        self.auto_lbl.setText(f"авто = {v}   [ (цена_прод + цена_зак) × лаг × 4 ]")

    def _set_auto(self):
        p={k:self.pw[k].value() for k in ['selling_price','order_unit_cost','lead_time']}
        self.pw['stockout_penalty'].setValue(auto_penalty(p))

    def _load(self):
        path,_=QFileDialog.getOpenFileName(self,"Выберите Excel файл","","Excel (*.xlsx *.xls)")
        if not path: return
        try:
            df=pd.read_excel(path,sheet_name="State_Input")
            df['yesterday_demand']=df['yesterday_demand'].replace([np.inf,-np.inf],np.nan).fillna(0).astype(int)
            df['day_of_year']=df['day_of_year'].fillna(1).astype(int)
            self.df=df; n=len(df); tr=int(n*0.6); te=n-tr
            self.file_lbl.setText(os.path.basename(path)); self.file_lbl.setStyleSheet(f"color:{ACCENT};")
            self.split_lbl.setText(f"  Всего: {n} дней  ·  обучение: {tr} дней  ·  тест: {te} дней  (60/40)")
            self.btn_start.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self,"Ошибка",f"Не удалось загрузить файл:\n{e}")

    def _start(self):
        if self.df is None: return
        p={k:w.value() for k,w in self.pw.items()}
        n=len(self.df); tr=int(n*0.6)
        self.go_train.emit(self.df.iloc[:tr].reset_index(drop=True),
                           self.df.iloc[tr:].reset_index(drop=True), p)

#  СТРАНИЦА 2: ОБУЧЕНИЕ

class TrainPage(QWidget):
    finished=pyqtSignal(object,object,object,object,list,list)
    go_home=pyqtSignal()

    def __init__(self):
        super().__init__(); self.worker=None; self._rewards=[]; self._epsilons=[]; self._build()

    def _build(self):
        root=QVBoxLayout(self); root.setSpacing(12); root.setContentsMargins(40,28,40,28)
        root.addWidget(page_header("ОБУЧЕНИЕ","тренировка DQN агента",cb_home=self._home))
        self.status_lbl=QLabel("Ожидание..."); self.status_lbl.setStyleSheet(f"color:{MUTED};")
        root.addWidget(self.status_lbl)
        self.pbar=QProgressBar(); self.pbar.setFixedHeight(12); root.addWidget(self.pbar)
        self.canvas=MplCanvas(figsize=(8,3)); root.addWidget(self.canvas)
        self.log=QTextEdit(); self.log.setReadOnly(True); self.log.setFixedHeight(130)
        root.addWidget(self.log)
        row=QHBoxLayout()
        self.btn_stop=QPushButton("■  ОСТАНОВИТЬ"); self.btn_stop.setObjectName("danger")
        self.btn_stop.clicked.connect(self._stop); row.addWidget(self.btn_stop); row.addStretch()
        root.addLayout(row); root.addStretch()

    def start(self,df_train,df_test,params):
        self.df_test=df_test; self.params=params
        ep=int(params['episodes']); self.pbar.setMaximum(ep); self.pbar.setValue(0)
        self._rewards.clear(); self._epsilons.clear(); self.log.clear()
        self.log.append(f"  Штраф за дефицит: {params['stockout_penalty']}")
        self.log.append(f"  Действия: 0 .. {int(params['capacity']//3//50)*50} (шаг 50)")
        self.log.append(f"  Эпох: {ep}"); self.log.append("─"*52)
        self.worker=TrainWorker(df_train,params,ep)
        self.worker.progress.connect(self._on_progress)
        self.worker.log_line.connect(self.log.append)
        self.worker.finished_ok.connect(lambda ag,rw,e: self._on_done(ag,df_train,rw,e))
        self.worker.start()

    def _on_progress(self,ep,reward,epsilon):
        self._rewards.append(reward); self._epsilons.append(epsilon)
        self.pbar.setValue(ep)
        self.status_lbl.setText(f"Эпизод {ep}  |  награда {reward:.0f}  |  ε {epsilon:.3f}")
        if ep%10==0: self._redraw()

    def _redraw(self):
        ax=self.canvas.ax(); ax.clear(); ax.set_facecolor(CARD)
        ax.plot(self._rewards,color=ACCENT,linewidth=0.8,alpha=0.6,label="Награда")
        if len(self._rewards)>20:
            sm=pd.Series(self._rewards).rolling(20).mean()
            ax.plot(sm,color=ACCENT2,linewidth=1.8,label="Среднее (20)")
        ax.legend(fontsize=11,labelcolor=TEXT,facecolor=CARD,edgecolor=BORDER)
        ax.set_xlabel("Эпизод",color=MUTED,fontsize=11); ax.set_ylabel("Награда",color=MUTED,fontsize=11)
        ax.set_title("Динамика обучения",color=TEXT,fontsize=12)
        ax.tick_params(colors=MUTED)
        for sp in ax.spines.values(): sp.set_color(BORDER)
        self.canvas.fig.set_facecolor(CARD); self.canvas.fig.tight_layout(); self.canvas.draw()

    def _on_done(self,agent,df_train,rewards,epsilons):
        self.log.append("─"*52); self.log.append("  ✓ ОБУЧЕНИЕ ЗАВЕРШЕНО"); self._redraw()
        self.finished.emit(agent,self.df_test,self.params,df_train,rewards,epsilons)

    def _stop(self):
        if self.worker: self.worker.stop()
        self.log.append("  ■ Остановлено пользователем.")

    def _home(self):
        if self.worker: self.worker.stop()
        self.go_home.emit()


#  СТРАНИЦА 3: ДАШБОРД

class DashPage(QWidget):
    go_home=pyqtSignal()

    def __init__(self):
        super().__init__()
        self.agent=self.env=self.params=self.state=None
        self.day_idx=0; self.cum_reward=self.cum_sold=self.cum_demand=0.0
        self.hist={k:[] for k in ['dates','inventory','demand','orders','sold','shortage','reward']}
        self._build()

    def _build(self):
        root=QVBoxLayout(self); root.setSpacing(0); root.setContentsMargins(0,0,0,0)
        self.tabs=QTabWidget(); root.addWidget(self.tabs)
        self.tabs.addTab(self._tab_daily(),   "ЕЖЕДНЕВНО")
        self.tabs.addTab(self._tab_charts(),  "ГРАФИКИ")
        self.tabs.addTab(self._tab_training(),"ОБУЧЕНИЕ")

    def _tab_header(self,title):
        row=QHBoxLayout(); w=QWidget(); w.setLayout(row)
        t=QLabel(title); t.setFont(QFont("Courier New",14,QFont.Bold))
        t.setStyleSheet(f"color:{ACCENT}; letter-spacing:3px;")
        row.addWidget(t); row.addStretch(); row.addWidget(home_btn(self._home))
        return w

    def _tab_daily(self):
        page=QWidget(); outer=QVBoxLayout(page); outer.setContentsMargins(0,0,0,0)
        scroll=QScrollArea(); scroll.setWidgetResizable(True); outer.addWidget(scroll)
        inner=QWidget(); lay=QVBoxLayout(inner); lay.setSpacing(14); lay.setContentsMargins(32,20,32,20)
        scroll.setWidget(inner)

        lay.addWidget(page_header("ЕЖЕДНЕВНЫЙ ПРОГНОЗ",cb_home=self._home))

        self.date_lbl=QLabel("—"); self.date_lbl.setFont(QFont("Courier New",18,QFont.Bold))
        self.date_lbl.setStyleSheet(f"color:{TEXT}; letter-spacing:3px;"); lay.addWidget(self.date_lbl)

        # рекомендация
        rec=QFrame()
        rec.setStyleSheet(f"background:{CARD}; border:2px solid {ACCENT}; border-radius:8px; padding:14px;")
        rl=QVBoxLayout(rec)
        t=QLabel("РЕКОМЕНДАЦИЯ АГЕНТА"); t.setStyleSheet(f"color:{MUTED}; font-size:13px;")
        rl.addWidget(t)
        self.rec_val=QLabel("—"); self.rec_val.setFont(QFont("Courier New",34,QFont.Bold))
        self.rec_val.setStyleSheet(f"color:{ACCENT};"); rl.addWidget(self.rec_val)
        self.rec_sub=QLabel(""); self.rec_sub.setStyleSheet(f"color:{MUTED}; font-size:13px;")
        rl.addWidget(self.rec_sub); lay.addWidget(rec)

        # ввод
        inp=QHBoxLayout()
        og=QGroupBox("ФАКТИЧЕСКИ ЗАКАЗАНО"); ol=QVBoxLayout(og)
        self.inp_order=QSpinBox(); self.inp_order.setRange(0,999999); self.inp_order.setSingleStep(50)
        ol.addWidget(self.inp_order); inp.addWidget(og)
        sg=QGroupBox("ФАКТИЧЕСКИ ПРОДАНО"); sl=QVBoxLayout(sg)
        self.inp_sold=QSpinBox(); self.inp_sold.setRange(0,999999); sl.addWidget(self.inp_sold)
        inp.addWidget(sg)
        self.btn_confirm=QPushButton("ПОДТВЕРДИТЬ ДЕНЬ  →")
        self.btn_confirm.setObjectName("primary"); self.btn_confirm.setFixedHeight(54)
        self.btn_confirm.clicked.connect(self._confirm); inp.addWidget(self.btn_confirm)
        lay.addLayout(inp)

        # карточки
        grid=QGridLayout(); grid.setSpacing(10)
        f,self._c_inv=info_card("ЗАПАС НА СКЛАДЕ",ACCENT3);     grid.addWidget(f,0,0)
        f,self._c_transit=info_card("ТОВАРОВ В ПУТИ",WARN);      grid.addWidget(f,0,1)
        f,self._c_demand=info_card("СПРОС ВЧЕРА",TEXT);          grid.addWidget(f,0,2)
        f,self._c_fill=info_card("УРОВЕНЬ ОБСЛУЖИВАНИЯ",ACCENT); grid.addWidget(f,1,0)
        f,self._c_dos=info_card("ДНЕЙ ЗАПАСА",ACCENT3);          grid.addWidget(f,1,1)
        f,self._c_reward=info_card("НАКОПЛ. ПРИБЫЛЬ",ACCENT);    grid.addWidget(f,1,2)
        lay.addLayout(grid)

        # товары в пути детально
        tg=QGroupBox("ТОВАРЫ В ПУТИ — ДЕТАЛЬНО"); tl=QVBoxLayout(tg)
        self.transit_lbl=QLabel("Нет заказов в пути")
        self.transit_lbl.setStyleSheet(f"color:{MUTED}; font-size:14px;")
        self.transit_lbl.setWordWrap(True); tl.addWidget(self.transit_lbl); lay.addWidget(tg)

        # прогноз
        fg=QGroupBox("ПРОГНОЗ СПРОСА НА 7 ДНЕЙ (сезонное среднее)"); fl=QVBoxLayout(fg)
        self.fc_canvas=MplCanvas(figsize=(8,2)); fl.addWidget(self.fc_canvas); lay.addWidget(fg)

        return page

    def _tab_charts(self):
        w=QWidget(); l=QVBoxLayout(w); l.setContentsMargins(16,16,16,16)
        l.addWidget(self._tab_header("ГРАФИКИ СИМУЛЯЦИИ"))
        self.ch_canvas=MplCanvas(figsize=(10,9),nrows=3); l.addWidget(self.ch_canvas)
        return w

    def _tab_training(self):
        w=QWidget(); l=QVBoxLayout(w); l.setContentsMargins(16,16,16,16)
        l.addWidget(self._tab_header("РЕЗУЛЬТАТЫ ОБУЧЕНИЯ"))
        self.tr_canvas=MplCanvas(figsize=(10,6),nrows=2); l.addWidget(self.tr_canvas)
        return w

    def setup(self,agent,df_test,params,df_train,train_rewards,train_epsilons):
        self.agent=agent; self.params=params
        self.env=InventoryEnv(df_test,params); self.state=self.env.reset()
        self.day_idx=0; self.cum_reward=self.cum_sold=self.cum_demand=0.0
        self.hist={k:[] for k in self.hist}
        self.btn_confirm.setEnabled(True)
        try: self.start_date=pd.to_datetime(df_test['date'].iloc[0])
        except: self.start_date=datetime(2024,1,1)
        self._render_training(train_rewards,train_epsilons)
        self._update_daily()

    def _cur_date(self): return self.start_date+timedelta(days=self.day_idx)

    def _update_daily(self):
        try:
            self._do_update_daily()
        except Exception:
            import traceback, sys
            print('ОШИБКА _update_daily:', traceback.format_exc(), file=sys.stderr)

    def _do_update_daily(self):
        if self.agent is None: return
        date=self._cur_date()
        self.date_lbl.setText(date.strftime("%A, %d %B %Y").upper())
        ai=self.agent.select_action(self.state,explore=False)
        rec=self.env.ACTIONS[ai]
        self.rec_val.setText(f"{rec} ед.")
        self.rec_sub.setText(
            f"лаг {self.params['lead_time']} дн.  ·  "
            f"штраф {self.params['stockout_penalty']}  ·  "
            f"запас {int(self.env.inv)} ед."
        )
        self.inp_order.setValue(rec)
        self._c_inv.setText(str(int(self.env.inv)))
        pending=list(self.env.pending_orders)
        self._c_transit.setText(str(int(sum(q for _,q in pending))))
        hist=list(self.env.demand_history)
        self._c_demand.setText(str(int(hist[-1])) if hist else "0")
        self._c_fill.setText(f"{self.cum_sold/self.cum_demand*100:.1f}%" if self.cum_demand>0 else "—")
        avg_d=max(1,np.mean(hist[-7:]) if hist else 1)
        self._c_dos.setText(f"{self.env.inv/avg_d:.1f} дн.")
        self._c_reward.setText(f"{self.cum_reward:,.0f}")
        if pending:
            self.transit_lbl.setText("\n".join(f"  → {qty} ед.  через {dl+1} дн." for dl,qty in sorted(pending)))
        else:
            self.transit_lbl.setText("  Нет заказов в пути")
        self._draw_forecast()

    def _draw_forecast(self):
        idx=min(self.env.day_idx,len(self.env.DAY_OF_YEAR)-1)
        d=self.env.DAY_OF_YEAR[idx]
        labels=[]; vals=[]
        for i in range(7):
            do=(d+i)%365; vals.append(self.env.seasonal_mean.get(do,self.env.demand_mean))
            labels.append((self._cur_date()+timedelta(days=i)).strftime("%a %d"))
        ax=self.fc_canvas.ax(); ax.clear(); ax.set_facecolor(CARD)
        ax.bar(labels,vals,color=ACCENT3,alpha=0.85)
        ax.set_ylabel("Спрос",color=MUTED,fontsize=11); ax.tick_params(colors=MUTED,labelsize=11)
        for sp in ax.spines.values(): sp.set_color(BORDER)
        self.fc_canvas.fig.set_facecolor(CARD); self.fc_canvas.fig.tight_layout(); self.fc_canvas.draw()

    def _confirm(self):
        try:
            self._do_confirm()
        except Exception:
            import traceback
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(self, 'Ошибка', traceback.format_exc())

    def _do_confirm(self):
        if self.agent is None or self.env.day_idx>=len(self.env.DEMAND): return
        ordered=self.inp_order.value()
        sold_in=self.inp_sold.value()
        self.env.DEMAND[self.env.day_idx] = sold_in

        ai=int(np.argmin([abs(ordered-a) for a in self.env.ACTIONS]))
        ns,reward,done,info=self.env.step(ai)
        self.cum_reward+=reward*100
        self.cum_sold+=info['sold']       
        self.cum_demand+=info['demand']   
        date=self._cur_date()
        for k,v in [('dates',date),('inventory',info['inventory']),('demand',info['demand']),
                    ('orders',info['order']),('sold',info['sold']),('shortage',info['shortage']),
                    ('reward',reward*100)]:
            self.hist[k].append(v)
        self.day_idx+=1; self._draw_charts()
        if done:
            self.btn_confirm.setEnabled(False)
            self.date_lbl.setText("СИМУЛЯЦИЯ ЗАВЕРШЕНА")
            self.rec_val.setText("Готово")
            self.rec_sub.setText(f"Итого дней: {self.day_idx}  ·  накопл. прибыль: {self.cum_reward:,.0f}")
            return
        self.state=ns; self._update_daily()

    def _draw_charts(self):
        h=self.hist
        if not h['dates']: return
        axes=self.ch_canvas.axes

        axes[0].clear(); axes[0].set_facecolor(CARD)
        axes[0].plot(h['dates'],h['inventory'],color=ACCENT3,label="Запас",linewidth=1.5)
        axes[0].plot(h['dates'],h['demand'],   color=ACCENT2,label="Спрос",linewidth=1,linestyle='--')
        axes[0].bar(h['dates'], h['orders'],   color=ACCENT, alpha=0.3,label="Заказы")
        axes[0].legend(fontsize=11,labelcolor=TEXT,facecolor=CARD,edgecolor=BORDER)
        axes[0].set_ylabel("Ед.",color=MUTED,fontsize=11)
        axes[0].set_title("Запас · Спрос · Заказы",color=TEXT,fontsize=11)

        axes[1].clear(); axes[1].set_facecolor(CARD)
        axes[1].bar(h['dates'],h['reward'],color=ACCENT,alpha=0.75)
        axes[1].set_ylabel("Прибыль",color=MUTED,fontsize=11)
        axes[1].set_title("Ежедневная прибыль",color=TEXT,fontsize=11)

        axes[2].clear(); axes[2].set_facecolor(CARD)
        cum=np.cumsum(h['shortage'])
        axes[2].fill_between(h['dates'],cum,color=ACCENT2,alpha=0.35)
        axes[2].plot(h['dates'],cum,color=ACCENT2,linewidth=1.5)
        axes[2].set_ylabel("Дефицит (нак.)",color=MUTED,fontsize=11)
        axes[2].set_title("Накопленный дефицит",color=TEXT,fontsize=11)

        for ax in axes:
            ax.tick_params(colors=MUTED,labelsize=10)
            for sp in ax.spines.values(): sp.set_color(BORDER)
            try: ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b')); ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            except: pass
        self.ch_canvas.fig.set_facecolor(CARD); self.ch_canvas.fig.tight_layout(); self.ch_canvas.draw()

    def _render_training(self,rewards,epsilons):
        axes=self.tr_canvas.axes; eps=list(range(1,len(rewards)+1))
        axes[0].clear(); axes[0].set_facecolor(CARD)
        axes[0].plot(eps,rewards,color=ACCENT,linewidth=0.8,alpha=0.5)
        if len(rewards)>20:
            sm=pd.Series(rewards).rolling(20).mean()
            axes[0].plot(eps,sm,color=ACCENT2,linewidth=1.8,label="Среднее (20)")
            axes[0].legend(fontsize=11,labelcolor=TEXT,facecolor=CARD,edgecolor=BORDER)
        axes[0].set_ylabel("Награда",color=MUTED); axes[0].set_title("Динамика обучения",color=TEXT)
        axes[1].clear(); axes[1].set_facecolor(CARD)
        axes[1].plot(eps,epsilons,color=WARN,linewidth=1.5)
        axes[1].set_ylabel("ε (эпсилон)",color=MUTED); axes[1].set_xlabel("Эпизод",color=MUTED)
        axes[1].set_title("Затухание исследования",color=TEXT)
        for ax in axes:
            ax.tick_params(colors=MUTED)
            for sp in ax.spines.values(): sp.set_color(BORDER)
        self.tr_canvas.fig.set_facecolor(CARD); self.tr_canvas.fig.tight_layout(); self.tr_canvas.draw()

    def _home(self): self.go_home.emit()


#  ГЛАВНОЕ ОКНО

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Управление запасами · DQN")
        self.setMinimumSize(1020,780); self.setStyleSheet(SS)
        self.stack=QStackedWidget(); self.setCentralWidget(self.stack)
        self.setup_page=SetupPage(); self.train_page=TrainPage(); self.dash_page=DashPage()
        self.stack.addWidget(self.setup_page)
        self.stack.addWidget(self.train_page)
        self.stack.addWidget(self.dash_page)
        self.setup_page.go_train.connect(self._go_train)
        self.train_page.finished.connect(self._go_dash)
        self.train_page.go_home.connect(self._confirm_home)
        self.dash_page.go_home.connect(self._confirm_home)

    def _go_train(self,df_train,df_test,params):
        self.stack.setCurrentIndex(1); self.train_page.start(df_train,df_test,params)

    def _go_dash(self,agent,df_test,params,df_train,rewards,epsilons):
        self.dash_page.setup(agent,df_test,params,df_train,rewards,epsilons)
        self.stack.setCurrentIndex(2)

    def _confirm_home(self):
        r=QMessageBox.question(self,"На главную",
            "Вернуться на главную страницу?\nТекущий прогресс будет сброшен.",
            QMessageBox.Yes|QMessageBox.No)
        if r==QMessageBox.Yes: self.stack.setCurrentIndex(0)

if __name__=="__main__":
    app=QApplication(sys.argv); app.setStyle("Fusion")
    win=MainWindow(); win.show(); sys.exit(app.exec_())