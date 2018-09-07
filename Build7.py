from pyomo.environ import *
import pyomo.opt as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%
def create_model(Qark, Qolst, H_NS, timestep = 3600, h0_nzk = -0.4, hmax_nzk = -0.3, hmin_nzk = -0.5, h0_mar = -0.2, hmax_mar = -0.1, hmin_mar = -0.3, h0_ijs = -0.2, hmax_ijs = -0.1, hmin_ijs = -0.3, Q0ijm_s = 0.0, Q0ijm_p = 0.0):
    model = AbstractModel()
    
    #declare parameters
    N = len(Qark) - 1 #prediction horizon
    model.T = RangeSet(0,N) #Timestep set
    dt = timestep #default is 3600 seconds
    A_nzk = 28.6e06 #surface area of NorthSea Canal in m2
    Hb_nzk = -15 #bottom level of NZK in m NAP
    S0_nzk = (h0_nzk-Hb_nzk)*A_nzk #Start storage of NZK
    #h0_nzk = -0.4  already in function parameters
    #Q0ijm_s = 0.0  already in function parameters
    #Q0ijm_p = 0.0  already in function parameters
    #hmax_nzk = hmax_nzk+0.2
    Smax_nzk = (hmax_nzk - Hb_nzk)*A_nzk
    Smin_nzk = (hmin_nzk - Hb_nzk)*A_nzk
    pe = 0.85 #pump efficiency
    
    Qmax_nzk = 260.0
    
    #declare variables
    model.Qijm_s = Var(model.T, within = NonNegativeReals, bounds=(0,500))
    model.Qijm_p = Var(model.T, within = NonNegativeReals, bounds=(0,Qmax_nzk))
    model.E = Var(model.T, within = NonNegativeReals, bounds=(0,None))
    model.S_nzk = Var(model.T, within = NonNegativeReals, bounds=(Smin_nzk,Smax_nzk))
    model.h_nzk = Var(model.T, bounds=(hmin_nzk, hmax_nzk))
    #model.is_downIJM = Var(model.T, within=Binary)
    
    Hb_mar = -15 #m NAP
    A_mar = 700e06 #surface area of markermeer in m2
    #hmax_mar = hmax_mar+0.2
    Smin_mar = (hmin_mar - Hb_mar)*A_mar
    Smax_mar = (hmax_mar - Hb_mar)*A_mar
    S0_mar = (h0_mar - Hb_mar)*A_mar
    
    model.Qor = Var(model.T, within = NonNegativeReals, bounds=(0, 200))
    model.S_mar = Var(model.T, within = NonNegativeReals, bounds=(Smin_mar,Smax_mar))
    model.h_mar = Var(model.T, bounds=(hmin_mar, hmax_mar))
    #model.is_downOR = Var(model.T, within = Binary)
    #hmin_ijs = hmin_ijs-0.5
    #hmax_ijs = hmax_ijs+0.6
    
    Hb_ijs = -15 #m NAP    
    A_ijs = 1133e06 #surface area of ijsselmeer in m2
    Smin_ijs = (hmin_ijs - Hb_ijs)*A_ijs
    Smax_ijs = (hmax_ijs - Hb_ijs)*A_ijs
    S0_ijs = (h0_ijs - Hb_ijs)*A_ijs
    
    model.Qhout = Var(model.T, within = NonNegativeReals, bounds=(0,1000))
    model.Qkorn = Var(model.T, within = NonNegativeReals, bounds=(0,1000))
    model.S_ijs = Var(model.T, within = NonNegativeReals, bounds=(Smin_ijs,Smax_ijs))
    model.h_ijs = Var(model.T, bounds=(hmin_ijs, hmax_ijs))
    model.Qden = Var(model.T, within = NonNegativeReals, bounds=(0,3000))
    #model.is_downHO = Var(model.T, within = Binary)
    
    model.s_den = Var(model.T, within = NonNegativeReals, bounds=(0,None))
    model.s_korn = Var(model.T, within = NonNegativeReals, bounds=(0,None))
    model.s_hout = Var(model.T, within = NonNegativeReals, bounds=(0,None))
    model.s_or = Var(model.T, within = NonNegativeReals, bounds=(0,None))
    model.s_ijm = Var(model.T, within = NonNegativeReals, bounds=(0,None))
    
    
    
    #model rules
    #mass balance
    def MassBalance_nzk(model, t):
        if t==0:
            return model.S_nzk[t] == S0_nzk
        return model.S_nzk[t] == model.S_nzk[t-1] + dt*(Qark[t-1] + model.Qor[t-1] - model.Qijm_s[t-1] - model.Qijm_p[t-1])
    model.MBnzk = Constraint(model.T, rule=MassBalance_nzk)
    
    def MassBalance_mar(model, t):
        if t==0:
            return model.S_mar[t] == S0_mar
        return model.S_mar[t] == model.S_mar[t-1] + dt*(model.Qhout[t-1] - model.Qor[t-1] - model.Qkorn[t-1])
    model.MBmar = Constraint(model.T, rule=MassBalance_mar)
    
    def MassBalance_ijs(model, t):
        if t==0:
            return model.S_ijs[t] == S0_ijs
        return model.S_ijs[t] == model.S_ijs[t-1] + dt*(Qolst[t-1] - model.Qhout[t-1] - model.Qden[t-1] + model.Qkorn[t-1])
    model.MBijs = Constraint(model.T, rule=MassBalance_ijs)
    
    #relation storage and water level
    def StorageWL_nzk(model,t):
        return model.h_nzk[t] == model.S_nzk[t]/A_nzk + Hb_nzk
    model.WLnzk = Constraint(model.T, rule=StorageWL_nzk)
    
    def StorageWL_mar(model,t):
        return model.h_mar[t] == model.S_mar[t]/A_mar + Hb_mar
    model.WLmar = Constraint(model.T, rule=StorageWL_mar)
    
    def StorageWL_ijs(model,t):
        return model.h_ijs[t] == model.S_ijs[t]/A_ijs + Hb_ijs
    model.WLijs = Constraint(model.T, rule=StorageWL_ijs)
    
    #energy consumption of ijmuiden
    def EnergyConsumption(model, t):
        return model.E[t] == dt * 1000 * 9.81 * model.Qijm_p[t] * (H_NS[t] - model.h_nzk[t]) / pe
    model.Energy = Constraint(model.T, rule=EnergyConsumption)
    
    #sluice behaviour
    g = 9.81  # m/s^2   gravitational acceleration
    
    #Ijmuiden constants
    N_ij = 7.0    # amount
    Qms_ij = 700.0 #sluice max discharge
    Qmp_ij = 260.0 #pump max discharge
    B_ij = 5.9  # m       width of sluice
    a_ij = 1.0  # none    sluice constant
    Hk_ij = 4.8 #m NAP   height of throat
    
    #Oranjesluizen constants
    N_or = 1.0    # amount
    Qms_or = 200.0 #sluice max discharge
    B_or = 4.0  # m       width of sluice
    a_or = 1.0  # none    sluice constant
    hbot_or = -4.5 #heigh of bottom in m NAP
    
    #Houtribsluizen constants
    N_hout = 8.0    # amount
    Qms_hout = 1000.0 #sluice max discharge
    B_hout = 18.0  # m       width of sluice
    a_hout = 0.95  # none    sluice constant
    hbot_hout = -4.5 #heigh of bottom in m NAP
    
    #Den Oever sluizen constants
    N_den = 15 + 10    # amount
    Qms_den = 3000.0 #sluice max discharge
    B_den = 12.0  # m       width of sluice
    a_den = 1.06  # none    sluice constant
    hbot_den = -4.7 #heigh of bottom in m NAP

#force sluice behaviour between 0 and a flexible max
    def Sluice_Ijmuiden(model, t):
        return ((model.Qijm_s[t]/(N_ij*a_ij*B_ij*Hk_ij))**2) <= (2*g*(model.h_nzk[t] - H_NS[t] + model.s_ijm[t])) #- M*(1 - model.is_downIJM[t]) <= 0
        #return 0 <= model.Qijm_s[t]**2/((N_ij*a_ij*B_ij*Hk_ij)**2 * (2*g*(model.h_nzk[t] - H_NS[t]))) <= 1
    model.slijm = Constraint(model.T, rule = Sluice_Ijmuiden)
    
    def Sluice_Ijmuiden2(model, t):
        return model.Qijm_s[t]/(N_ij*a_ij*B_ij*Hk_ij) >= 0
    model.slijm2 = Constraint(model.T, rule = Sluice_Ijmuiden2)
    
    def Sluice_Oranjesluizen(model, t):
        return (model.Qor[t]/(N_or*a_or*B_or*(model.h_nzk[t] - hbot_or)))**2 <= (2*g*(model.h_mar[t] - model.h_nzk[t] + model.s_or[t]))
    model.slor = Constraint(model.T, rule = Sluice_Oranjesluizen)
    
    def Sluice_Oranjesluizen2(model, t):
        return model.Qor[t]/(N_or*a_or*B_or*(model.h_nzk[t] - hbot_or)) >= 0
    model.slor2 = Constraint(model.T, rule = Sluice_Oranjesluizen2)
    
    def Sluice_Houtribsluis(model, t):
        return (model.Qhout[t]/(N_hout*a_hout*B_hout*(model.h_mar[t] - hbot_hout)))**2 <= (2*g*(model.h_ijs[t] - model.h_mar[t] + model.s_hout[t]))
    model.slho = Constraint(model.T, rule=Sluice_Houtribsluis)
    
    def Sluice_Houtribsluis2(model, t):
        return model.Qhout[t]/(N_hout*a_hout*B_hout*(model.h_mar[t] - hbot_hout)) >= 0
    model.slho2 = Constraint(model.T, rule=Sluice_Houtribsluis2)
    
    def Sluice_Kornwederzand(model, t):
        return (model.Qkorn[t]/(N_hout*a_hout*B_hout*(model.h_ijs[t] - hbot_hout)))**2 <= (2*g*(model.h_mar[t] - model.h_ijs[t] + model.s_korn[t]))
    model.slko = Constraint(model.T, rule=Sluice_Kornwederzand)
    
    def Sluice_Kornwederzand2(model, t):
        return model.Qkorn[t]/(N_hout*a_hout*B_hout*(model.h_ijs[t] - hbot_hout)) >= 0
    model.slko2 = Constraint(model.T, rule=Sluice_Kornwederzand2)
    
    def Sluice_DenOever(model, t):
        return (model.Qden[t]/(N_den*a_den*B_den*(H_NS[t] - hbot_den)))**2 <= (2*g*(model.h_ijs[t] - H_NS[t] + model.s_den[t]))
    model.slde = Constraint(model.T, rule=Sluice_DenOever)
    
    def Sluice_DenOever2(model, t):
        return model.Qden[t]/(N_den*a_den*B_den*(H_NS[t] - hbot_den)) >= 0
    model.slde2 = Constraint(model.T, rule=Sluice_DenOever2)
    
    def Slack_Ijmuiden(model, t):
        return model.h_nzk[t] - H_NS[t] + model.s_ijm[t] >= 0
    model.slackijm = Constraint(model.T, rule = Slack_Ijmuiden)
    
    def Slack_Oranje(model, t):
        return model.h_mar[t] - model.h_nzk[t] + model.s_or[t] >= 0
    model.slackor = Constraint(model.T, rule = Slack_Oranje)
    
    def Slack_Hout(model, t):
        return model.h_ijs[t] - model.h_mar[t] + model.s_hout[t] >= 0
    model.slackhout = Constraint(model.T, rule = Slack_Hout)
    
    def Slack_Korn(model, t):
        return model.h_mar[t] - model.h_ijs[t] + model.s_korn[t] >= 0
    model.slackkorn = Constraint(model.T, rule = Slack_Korn)
    
    def Slack_Den(model, t):
        return model.h_ijs[t] - H_NS[t] + model.s_den[t] >= 0
    model.slackden = Constraint(model.T, rule = Slack_Den)
    
    #Objectives
    def obj(model):
        Escale = (N+1)*1000*9.81*3*260*dt/pe
        def minEnergy():
            return (sum([model.E[t] for t in model.T])/Escale)**2
        def slacks():
            return (sum([model.s_ijm[t] + 10*model.s_or[t] + 100*model.s_hout[t] + 10*model.s_korn[t] + model.s_den[t] for t in model.T]))
        return minEnergy() + slacks()
    model.obj = Objective(rule=obj, sense=minimize)
    
    
    return model

#%%
Q_ark = pd.read_csv(r'D:\Users\Heijden\Documents\ModelPyomo\Q_ARK_MAARSSEN.csv', index_col = 'index')
Q_ark.index = pd.to_datetime(Q_ark.index)
H_NS = np.linspace(-1, 0.5, 11)
H_NS = np.append(H_NS, np.flip(H_NS, 0))
H_NS = np.append(H_NS, H_NS)

Q_olst = pd.read_csv(r'D:\Users\Heijden\Documents\Model RTCtools\data\olst debiet.csv', sep=';', usecols=['datum','tijd','waarde'])
d=[]
d.append([str(Q_olst.loc[i, 'datum'])+' '+str(Q_olst.loc[i, 'tijd']) for i in Q_olst.index])
#print(pd.to_datetime(d[0]))
Q_olst.index=pd.to_datetime(d[0])
Q_olst.drop('datum',axis=1)
Q_olst.drop('tijd',axis=1)

tnow = pd.to_datetime('2017-01-01 00:00:00')
ttil = tnow + pd.Timedelta('1D')

ind1 = pd.date_range(tnow, ttil, freq='1H')
Qa = []
H = []
Qo = []
j = 0
for i in ind1:
    Qa.append((Q_ark.loc[(Q_ark.index.year == i.year) & (Q_ark.index.month == i.month) & (Q_ark.index.day == i.day),'Q'][0]))
    H.append(H_NS[j])
    Qo.append((Q_olst.loc[(Q_olst.index.year == i.year) & (Q_olst.index.month == i.month) & (Q_olst.index.day == i.day),'waarde'][0]))
    j = j+1
#Q.index = ind
#print(ind)
ind2 = np.arange(0,25,1)
Qark1 = pd.DataFrame(data = {'Q': Qa})
Qark1.index = ind2
Qolst1 = pd.DataFrame(data = {'Q': Qo})
Qolst1.index = ind2
Hns1 = pd.DataFrame(data = {'H': H})
Hns1.index = ind2
Qark = Qark1.to_dict()['Q']
Hns = Hns1.to_dict()['H']
Qolst = Qolst1.to_dict()['Q']
#%%
model = create_model(Qark = Qark, H_NS = Hns, Qolst = Qolst)
instance = model.create_instance()
#%%
opt = SolverFactory('ipopt')
#opt.options['max_iter'] = 6000
opt.options["halt_on_ampl_error"] = 'yes'
#opt.options['constr_viol_tol'] = 0.00000000001
#opt.options['dual_inf_tol'] = 0.000001
#opt.options['bound_relax_factor'] = 0
#opt.options['send_statuses'] = False
#opt.options['acceptable_constr_viol_tol'] = 0.000001
for t in instance.T:
    instance.h_nzk[t] = -0.35
    instance.h_mar[t] = -0.25
    instance.h_ijs[t] = -0.15
results = opt.solve(instance, tee=True, logfile = 'log1.log', symbolic_solver_labels=True)#, coldstart=True)#, warmstart=True)
#%%
E = []
Snzk = []
Qijmp = []
Qijms = []
hnzk = []
#bijm = []
Smar = []
Qor = []
hmar = []
#bor = []
Sijs = []
Qhout = []
#bho = []
hijs = []
Qden = []
Qkorn = []
sijm = []
sor = []
shout = []
skorn = []
sden = []

for i in instance.T:
    E.append(instance.E[i].value)
    Snzk.append(instance.S_nzk[i].value)
    hnzk.append(instance.h_nzk[i].value)
    Qijmp.append(instance.Qijm_p[i].value)
    Qijms.append(instance.Qijm_s[i].value)
    #bijm.append(instance.is_downIJM[i].value)
    Smar.append(instance.S_mar[i].value)
    Qor.append(instance.Qor[i].value)
    hmar.append(instance.h_mar[i].value)
    #bor.append(instance.is_downOR[i].value)
    Sijs.append(instance.S_ijs[i].value)
    Qhout.append(instance.Qhout[i].value)
    #bho.append(instance.is_downHO[i].value)
    hijs.append(instance.h_ijs[i].value)
    Qden.append(instance.Qden[i].value)
    Qkorn.append(instance.Qkorn[i].value)
    sijm.append(instance.s_ijm[i].value)
    sor.append(instance.s_or[i].value)
    shout.append(instance.s_hout[i].value)
    skorn.append(instance.s_korn[i].value)
    sden.append(instance.s_den[i].value)
#%%
plt.figure()
fig, ax1 = plt.subplots()
ax1.set_xlabel('Datetime')
ax1.set_ylabel('Water level [m NAP]')
ax1.step(ind1,hnzk, label='NZK WL')
ax1.step(ind1,Hns1,'--b', label='NS WL')
ax1.legend(loc=2)
ax2 = ax1.twinx()
ax2.set_ylabel('Discharge [m3/s]')
ax2.step(ind1, Qijms, 'k', label='Qsluice')
ax2.step(ind1, Qark1, 'red', label='Qark')
ax2.step(ind1, Qor, 'orange', label='Qoranje')
ax2.step(ind1, Qijmp, 'g', label='Qpump')
ax2.legend(loc=1)

#plt.figure()
#plt.step(ind1, bijm)
#%%
plt.figure()
fig, ax1 = plt.subplots()
ax1.set_xlabel('Datetime')
ax1.set_ylabel('Water level [m NAP]')
ax1.step(ind1,hmar, label='Markermeer WL')
ax1.step(ind1,hnzk,'--b', label='NZK WL')
ax1.legend(loc=2)
ax2 = ax1.twinx()
ax2.set_ylabel('Discharge [m3/s]')
ax2.step(ind1, Qor, 'g', label='Qoranjesluizen')
ax2.step(ind1, Qhout, 'orange',label='Qhoutrib')
ax2.step(ind1, Qkorn, 'purple', label='Qkornwederzand')
#ax2.step(ind1, Qs, 'k', label='Qsluice')
#ax2.step(ind1, Qark1, 'red', label='Qin')
ax2.legend(loc=1)

#plt.figure()
#plt.step(ind1, bor)


#%%

plt.figure()
fig, ax1 = plt.subplots()
ax1.set_xlabel('Datetime')
ax1.set_ylabel('Water level [m NAP]')
ax1.step(ind1,hijs, label='Ijsselmeer WL')
ax1.step(ind1,hmar,'--b', label='Markermeer WL')
ax1.legend(loc=2)
ax2 = ax1.twinx()
ax2.set_ylabel('Discharge [m3/s]')
ax2.step(ind1, Qolst1, 'g', label='Qolst')
ax2.step(ind1, Qhout, 'orange',label='Qhoutrib')
ax2.step(ind1, Qkorn, 'purple', label='Qkornwederzand')
#ax2.step(ind1, Qs, 'k', label='Qsluice')
#ax2.step(ind1, Qark1, 'red', label='Qin')
ax2.legend(loc=1)

#plt.figure()
#plt.step(ind1, bho)


#%%
plt.figure()
fig, ax1 = plt.subplots()
ax1.set_xlabel('Datetime')
ax1.set_ylabel('Water level [m NAP]')
ax1.step(ind1,hijs, label='Ijsselmeer WL')
ax1.step(ind1,Hns1,'--b', label='Waddenzee WL')
ax1.legend(loc=2)
ax2 = ax1.twinx()
ax2.set_ylabel('Discharge [m3/s]')
ax2.step(ind1, Qolst1, 'g', label='Qolst')
ax2.step(ind1, Qden, 'k', label='Qdenoever')
#ax2.step(ind1, Qs, 'k', label='Qsluice')
#ax2.step(ind1, Qark1, 'red', label='Qin')
ax2.legend(loc=1)

#%%
#MB_nzk = []
MB_mar = []
MB_ijs = []
sl_ijm = []
sl_or = []
sl_hout = []
sl_korn = []
sl_den = []

def MB_calc(S, Qin, Qout):
    c = []
    for i in range(1,len(S)):
        c.append(S[i] - S[i-1] - 3600*(Qin[i-1] - Qout[i-1]))
    return c

def Sl_calc(Q, hi, ho, s, N, a, B, hb):
    g = 9.81
    c1 = []
    c2 = []
    for i in range(len(Q)):
        c1.append((Q[i]/(N*a*B*(ho[i] - hb)))**2)
        c2.append((2*g*(hi[i] - ho[i] + s[i])))
    return c1, c2

def Sl_ijm(Q, hi, ho, s, N, a, B, Hk):
    g = 9.81
    c1 = []
    c2 = []
    for i in range(len(Q)):
        c1.append((Q[i]/(N*a*B*Hk))**2)
        c2.append((2*g*(hi[i] - ho[i] + s[i])))
    return c1, c2

def h_calc(h, S, A):
    hb = -15
    c = []
    for i in range(len(h)):
        c.append(h[i] - S[i]/A - hb)
    return c

Qinnzk = Qark1.Q + Qor

MB_nzk = MB_calc(Snzk, Qark1.Q + Qor, Qijms + Qijmp)
MB_mar = MB_calc(Smar, Qhout, Qkorn + Qor)
MB_ijs = MB_calc(Sijs, Qolst1.Q + Qkorn, Qden + Qhout)
H_nzk = h_calc(hnzk, Snzk, 28.6e06)
H_mar = h_calc(hmar, Smar, 700e06)
H_ijs = h_calc(hijs, Sijs, 1133e06)


plt.figure()
plt.plot(MB_nzk)
plt.title('Mass Balance NZK')
plt.figure()
plt.plot(MB_mar)
plt.title('Mass Balance MAR')
plt.figure()
plt.plot(MB_ijs)
plt.title('Mass Balance IJS')
plt.figure()
plt.plot(H_nzk)
plt.title('Water level - storage relationship NZK')
plt.figure()
plt.plot(H_mar)
plt.title('Water level - storage relationship MAR')
plt.figure()
plt.plot(H_ijs)
plt.title('Water level - storage relationship IJS')

#sluice behaviour
g = 9.81  # m/s^2   gravitational acceleration
    
    #Ijmuiden constants
N_ij = 7.0    # amount
Qms_ij = 700.0 #sluice max discharge
Qmp_ij = 260.0 #pump max discharge
B_ij = 5.9  # m       width of sluice
a_ij = 1.0  # none    sluice constant
Hk_ij = 4.8 #m NAP   height of throat
    
    #Oranjesluizen constants
N_or = 1.0    # amount
Qms_or = 200.0 #sluice max discharge
B_or = 4.0  # m       width of sluice
a_or = 1.0  # none    sluice constant
hbot_or = -4.5 #heigh of bottom in m NAP
    
    #Houtribsluizen constants
N_hout = 8.0    # amount
Qms_hout = 1000.0 #sluice max discharge
B_hout = 18.0  # m       width of sluice
a_hout = 0.95  # none    sluice constant
hbot_hout = -4.5 #heigh of bottom in m NAP
    
    #Den Oever sluizen constants
N_den = 15 + 10    # amount
Qms_den = 3000.0 #sluice max discharge
B_den = 12.0  # m       width of sluice
a_den = 1.06  # none    sluice constant
hbot_den = -4.7 #heigh of bottom in m NAP

sl_ijm1, sl_ijm2 = Sl_ijm(Qijms, hnzk, Hns1.H, sijm, N_ij, a_ij, B_ij, Hk_ij)
sl_or1, sl_or2 = Sl_calc(Qor, hmar, hnzk, sor, N_or, a_or, B_or, hbot_or)
sl_hout1, sl_hout2 = Sl_calc(Qhout, hijs, hmar, shout, N_hout, a_hout, B_hout, hbot_hout)
sl_korn1, sl_korn2 = Sl_calc(Qkorn, hmar, hijs, skorn, N_hout, a_hout, B_hout, hbot_hout)
sl_den1, sl_den2 = Sl_calc(Qden, hijs, Hns1.H, sden, N_den, a_den, B_den, hbot_den)

plt.figure()
plt.plot(sl_ijm1, 'g')
plt.plot(sl_ijm2, 'r')
plt.title('Sluice Ijmuiden, c:g<=r')
plt.figure()
plt.plot(sl_or1, 'g')
plt.plot(sl_or2, 'r')
plt.title('Sluice Oranje, c:g<=r')
plt.figure()
plt.plot(sl_hout1, 'g')
plt.plot(sl_hout2, 'r')
plt.title('Sluice Hout, c:g<=r')
plt.figure()
plt.plot(sl_korn1, 'g')
plt.plot(sl_korn2, 'r')
plt.title('Sluice Korn, c:g<=r')
plt.figure()
plt.plot(sl_den1, 'g')
plt.plot(sl_den2, 'r')
plt.title('Sluice Den Oever, c:g<=r')

plt.figure()
plt.plot(sijm)
plt.title('Slack Ijmuiden')
plt.figure()
plt.plot(sor)
plt.title('Slack Oranje')
plt.figure()
plt.plot(shout)
plt.title('Slack Hout')
plt.figure()
plt.plot(skorn)
plt.title('Slack Korn')
plt.figure()
plt.plot(sden)
plt.title('Slack Den Oever')





#%%
#instance.pprint()




#%%
"""
Created on Fri Aug 31 15:13:28 2018

@author: heijden
"""

