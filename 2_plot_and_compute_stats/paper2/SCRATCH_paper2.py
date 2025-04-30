# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 15:09:38 2025

@author: hanna
"""
#%%
"""
Figure 3
"""

"""
"Meta-Learning" (across sessions)


[1] FIRST

"""


dColor = {50: yellow, -50:orange, 90:purple, -90:blue} #check colors

BEH_KEY = 'time'

mode = 'rotation'
open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
dDates, dDegs, dDegs2 = pickle.load(open_file)
open_file.close()

mode = 'shuffle'
open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
dDates_shuffle, dDegs_shuffle, dDegs2_shuffle = pickle.load(open_file)
open_file.close()

fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(10,5))


for j, mode in enumerate(['rotation', 'shuffle']):
    for i in [0,1]:
    
        subj, subject = dSubject[i]
    
        open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
        dDates, dDegs, _ = pickle.load(open_file)
        open_file.close()
        
        date_list_ = [datetime.strptime(date_str, "%Y%m%d") for date_str in dDates[i]]
        
        
        if mode == 'rotation':
            #color_list_ = [orange if deg==50 else purple for deg in dDegs[i]]
            color_list_ = [dColor[deg] for deg in dDegs2[i]]
            dfT = pd.DataFrame({'deg': dDegs2[i],'date': date_list_, 'color': color_list_, 'VAR': dBEH[mode][i][BEH_KEY][:,0]}).sort_values(by=['date']).reset_index()
      
        else:
            color_list_ = [magenta]*len(date_list_)
            dfT = pd.DataFrame({'date': date_list_, 'color': color_list_, 'VAR': dBEH[mode][i][BEH_KEY][:,0]}).sort_values(by=['date']).reset_index()
      
            
        

        axis = ax[j][i]
        
        xdata = np.arange(len(dfT))
        ydata = dfT['VAR'].values
        axis.scatter(xdata, ydata, c=dfT['color'].values, edgecolor='k')
        axis.plot(dfT['VAR'].values, color='grey', lw=0.75, zorder=0)
        
        axis.axhline(0, color='grey', zorder=0, ls='--')
        
        count=0
        if mode == 'rotation':
            
            if i == 0:
                deg_order = [50,90,-50,-90]
            else:
                deg_order = [-50,-90,50,90]
            
            for deg in deg_order:#[50,90,-50,-90]:#np.unique(dfT['deg'].values):
            
                dfT_deg = dfT.loc[dfT['deg']==deg]
                
                xdata = np.arange(count,len(dfT_deg)+count)
                ydata = dfT_deg['VAR'].values

            
            
            
                popt, pcov = curve_fit(linear_model, xdata, ydata)
                
                # Calculate degrees of freedom
                n = len(ydata)  # Number of data points
                k = len(popt)   # Number of parameters
                df = n - k
                
                standard_errors = np.sqrt(np.diag(pcov))
                t_values = popt / standard_errors
                p_values = 2 * (1 - stats.t.cdf(np.abs(t_values), df))
                
                print(f'{i}| time  | {deg} | m={popt[0]:.2f} p={p_values[0]:.2f}')
                
                
                slope, intercept = popt
                
                x1 = count#-0.5
                x2 = len(xdata)+count#n+0.5
                
                y1 = slope*x1 + intercept
                y2 = slope*x2 + intercept
                
                if deg == -50:
                    color=orange
                elif deg == 50:
                    color=yellow
                elif deg == -90:
                    color=blue
                else:
                    color=purple
                
                axis.plot([x1,x2],[y1,y2], color=color, lw=4, ls='-', zorder=10)
                    
                count+=len(xdata)
                
                if p_values[0] < 0.05:
                    axis.text((x2+x1)/2, -75, 'p<0.05', color=color)
                    print('TEST')
        
        elif mode == 'shuffle':

                popt, pcov = curve_fit(linear_model, xdata, ydata)
                
                # Calculate degrees of freedom
                n = len(ydata)  # Number of data points
                k = len(popt)   # Number of parameters
                df = n - k
                
                standard_errors = np.sqrt(np.diag(pcov))
                t_values = popt / standard_errors
                p_values = 2 * (1 - stats.t.cdf(np.abs(t_values), df))
                
                print(f'{i}| time | m={popt[0]:.2f} p={p_values[0]:.2f}')
                
                
                slope, intercept = popt
                
                x1 = -0.5
                x2 = len(xdata)
                
                y1 = slope*x1 + intercept
                y2 = slope*x2 + intercept
     
                
                axis.plot([x1,x2],[y1,y2], color='k', lw=4, ls='-', zorder=10)
                    
                if p_values[0] < 0.05:
                    axis.text((x2+x1)/2, -75, 'p<0.05', color=color)
                    

                

ax[0][0].set_title('Monkey A | rotation', loc='left')
ax[1][0].set_title('Monkey A | shuffle', loc='left')

ax[0][1].set_title('Monkey B | rotation', loc='left')
ax[1][1].set_title('Monkey B | shuffle', loc='left')

fig.text(0.5,0,'session number (ordered by date)', ha='center', fontsize=fontsize+2)
fig.subplots_adjust(wspace=0.05, hspace=0.3, bottom=0.1)


ax[0][0].set_ylim([-100,400])
ax[0][0].set_yticks(np.arange(-100,401,100))
ax[0][0].set_yticklabels([-100, 'BL', 100, 200, 300, 400])




ax[0][0].set_ylabel('performance\n(% Δ in time from BL)')
ax[1][0].set_ylabel('performance\n(% Δ in time from BL)')


legend_elements = [Line2D([0],[0], marker='o', markersize=10, color=yellow, lw=0, markeredgecolor='w', label='+50'),
                   Line2D([0],[0], marker='o', markersize=10, color=orange, lw=0, markeredgecolor='w', label='-50'),
                   Line2D([0],[0], marker='o', markersize=10, color=purple, lw=0, markeredgecolor='w', label='+90'),
                   Line2D([0],[0], marker='o', markersize=10, color=blue, lw=0, markeredgecolor='w', label='-90'),]
    

ax[0][0].legend(handles=legend_elements, ncol=2, fontsize=fontsize, columnspacing=0.3, handletextpad=0.0, loc=(0.65,0.75))#'upper right')
ax[0][1].legend(handles=legend_elements, ncol=2, fontsize=fontsize, columnspacing=0.3, handletextpad=0.0, loc=(0.65,0.75))#'upper right')





"""
"""


#%%

"""
"Meta-Learning" (across sessions)


[2] initial rate


"""

BEH_KEY = 'time'

mode = 'rotation'
open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
dDates, dDegs, dDegs2 = pickle.load(open_file)
open_file.close()

mode = 'shuffle'
open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
dDates_shuffle, dDegs_shuffle, dDegs2_shuffle = pickle.load(open_file)
open_file.close()

fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(10,5))


for j, mode in enumerate(['rotation', 'shuffle']):
    for i in [0,1]:
    
        subj, subject = dSubject[i]
    
        open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
        dDates, dDegs, _ = pickle.load(open_file)
        open_file.close()
        
        date_list_ = [datetime.strptime(date_str, "%Y%m%d") for date_str in dDates[i]]
        
        
        if mode == 'rotation':
            color_list_ = [orange if deg==50 else purple for deg in dDegs[i]]
            
        else:
            color_list_ = [magenta]*len(date_list_)
      
            
            
        v0 = dBEH[mode][i][BEH_KEY][:,0]
        v1 = dBEH[mode][i][BEH_KEY][:,1]
        
        mean_delta = np.abs(v1-v0)#np.mean(np.diff(dBEH[mode][i][BEH_KEY], axis=1), axis=1) #v1-v0
        
        dfT = pd.DataFrame({'date': date_list_, 'color': color_list_, 'VAR': mean_delta})
        
        if mode == 'rotation':
            dfT['deg'] = dDegs2[i]
       
        dfT = dfT.sort_values(by=['date']).reset_index()
      
           
      
        axis = ax[j][i]
        
        xdata = np.arange(len(dfT))
        ydata = dfT['VAR'].values
        axis.scatter(xdata, ydata, c=dfT['color'].values, edgecolor='k')
        axis.plot(dfT['VAR'].values, color='grey', lw=0.75, zorder=0)
        
        axis.axhline(0, color='grey', zorder=0, ls='--')
        
        count=0
        if mode == 'rotation':
            
            if i == 0:
                deg_order = [50,90,-50,-90]
            else:
                deg_order = [-50,-90,50,90]
            
            for deg in deg_order:#[50,90,-50,-90]:#np.unique(dfT['deg'].values):
            
                dfT_deg = dfT.loc[dfT['deg']==deg]
                
                xdata = np.arange(count,len(dfT_deg)+count)
                ydata = dfT_deg['VAR'].values

            
            
            
                popt, pcov = curve_fit(linear_model, xdata, ydata)
                
                # Calculate degrees of freedom
                n = len(ydata)  # Number of data points
                k = len(popt)   # Number of parameters
                df = n - k
                
                standard_errors = np.sqrt(np.diag(pcov))
                t_values = popt / standard_errors
                p_values = 2 * (1 - stats.t.cdf(np.abs(t_values), df))
                
                print(f'{i}| time  | {deg} | m={popt[0]:.2f} p={p_values[0]:.2f}')
                
                
                slope, intercept = popt
                
                x1 = count#-0.5
                x2 = len(xdata)+count#n+0.5
                
                y1 = slope*x1 + intercept
                y2 = slope*x2 + intercept
                
                if deg == -50:
                    color=orange
                elif deg == 50:
                    color=yellow
                elif deg == -90:
                    color=blue
                else:
                    color=purple
                
                axis.plot([x1,x2],[y1,y2], color=color, lw=4, ls='-', zorder=10)
                    
                count+=len(xdata)
                
                if p_values[0] < 0.05:
                    axis.text((x2+x1)/2, -75, 'p<0.05', color=color)
                    print('TEST')
        
        elif mode == 'shuffle':

                popt, pcov = curve_fit(linear_model, xdata, ydata)
                
                # Calculate degrees of freedom
                n = len(ydata)  # Number of data points
                k = len(popt)   # Number of parameters
                df = n - k
                
                standard_errors = np.sqrt(np.diag(pcov))
                t_values = popt / standard_errors
                p_values = 2 * (1 - stats.t.cdf(np.abs(t_values), df))
                
                print(f'{i}| time | m={popt[0]:.2f} p={p_values[0]:.2f}')
                
                
                slope, intercept = popt
                
                x1 = -0.5
                x2 = len(xdata)
                
                y1 = slope*x1 + intercept
                y2 = slope*x2 + intercept
     
                
                axis.plot([x1,x2],[y1,y2], color='k', lw=4, ls='-', zorder=10)
                    
                if p_values[0] < 0.05:
                    axis.text((x2+x1)/2, -75, 'p<0.05', color=color)
                    


legend_elements = [Line2D([0],[0], marker='o', markersize=10, color=orange, lw=0, markeredgecolor='w', label='easy'),
                   Line2D([0],[0], marker='o', markersize=10, color=purple, lw=0, markeredgecolor='w', label='hard')]
    

ax[0][0].legend(handles=legend_elements, ncol=1, fontsize=fontsize, columnspacing=0.3, handletextpad=0.0, loc=(0.65,0.75))#'upper right')
ax[0][1].legend(handles=legend_elements, ncol=1, fontsize=fontsize, columnspacing=0.3, handletextpad=0.0, loc=(0.65,0.75))#'upper right')




ax[0][0].set_title('Monkey A | rotation', loc='left')
ax[1][0].set_title('Monkey A | shuffle', loc='left')

ax[0][1].set_title('Monkey B | rotation', loc='left')
ax[1][1].set_title('Monkey B | shuffle', loc='left')

fig.text(0.5,0,'session number (ordered by date)', ha='center', fontsize=fontsize+2)
fig.subplots_adjust(wspace=0.05, hspace=0.3, bottom=0.1)



# ax[0][0].set_ylabel('Δ performance\n (%; second-first)')
# ax[1][0].set_ylabel('Δ performance\n (%; second-first)')


ax[0][0].set_ylabel('mean Δ performance \nper trial set')
ax[1][0].set_ylabel('mean Δ performance \nper trial set')



#%%

#%%
#%%


# #20250307


# mode = 'rotation'
# open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
# dDates, dDegs, dDegs2 = pickle.load(open_file)
# open_file.close()

# mode = 'shuffle'
# open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
# dDates_shuffle, dDegs_shuffle, dDegs2_shuffle = pickle.load(open_file)
# open_file.close()







# # import warnings
# # warnings.filterwarnings("ignore")


# BEH_KEY = 'dist'

# mode = 'rotation'
# open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
# dDates, dDegs, dDegs2 = pickle.load(open_file)
# open_file.close()

# mode = 'shuffle'
# open_file = open(os.path.join(pickle_path, f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
# dDates_shuffle, dDegs_shuffle, dDegs2_shuffle = pickle.load(open_file)
# open_file.close()

# fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(6,6))

# for i in [0,1]:
    
#     subj, subject = dSubject[i]


#     ind50 = np.where(dDegs[i]==50)[0]
#     ind90 = np.where(dDegs[i]==90)[0]

#     xdata = [50]*len(ind50) + [90]*len(ind90) + [4]*len(dDates_shuffle[i])
#     ydata = []

    
#     'AoL'
#     for ind in ind50:
#         v = dBEH['rotation'][i][BEH_KEY][ind,:]
#         aol_ = v[0]#(v[0]-np.min(v))/v[0]
#         ydata.append(aol_)
    
#     for ind in ind90:
#         v = dBEH['rotation'][i][BEH_KEY][ind,:]
#         aol_ = v[0]#(v[0]-np.min(v))/v[0]
#         ydata.append(aol_)
    
#     for d in range(len(dDates_shuffle[i])):
#         v = dBEH['shuffle'][i][BEH_KEY][d,:]
#         aol_ = v[0]#v[0]-np.min(v))/v[0]
#         ydata.append(aol_)
        

#     print(len(xdata), len(ydata))


#     sb.swarmplot(x=xdata, y=ydata, order=[50,90,4], palette={50:orange, 90:purple, 4:'grey'},ax=ax[i])
#     xdata = np.array(xdata)
#     ydata = np.array(ydata)
    
#     for KEY in [50,90,4]:
#         v1 = ydata[np.where(xdata==KEY)[0]]

#         t,p,star = compute_one_sample_ttest(v1,1)
#         n1 = len(v1)
#         print(f'{subject}: {KEY}| t{n1-1}={t:.2f}, p={p:2e} ({star})')
# # plt.boxplot(ydata, showfliers=False)
# # print(compute_one_sample_ttest(ydata,1))





#%%

"""
SINGLE SUBJECT

EXPONENTIAL - combine +/- for rotation

"""

fontsize = 35
mpl.rcParams["font.size"] = fontsize

fig, ax = plt.subplots(nrows=1,ncols=1,sharex=True,sharey=False,figsize=(18,18))
#fig.text(0,0.5, '% Δ from BL', ha='center', va='center', fontsize=fontsize+2, rotation=90)
fig.subplots_adjust(wspace=0.05, hspace=0.3, left=0.05)

mode = 'rotation'
open_file = open(os.path.join(root_path, 'pickles', f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
dDates, dDegs, dDegs2 = pickle.load(open_file)
open_file.close()


ANOVA_INDS = []
ANOVA_RES  = []

for i in [0]:
    
    subj, subject = dSubject[i]
    
    ind50 = np.where(dDegs[i]==50)[0]
    ind90 = np.where(dDegs[i]==90)[0]
    
    ax1 = ax#[i]

    ax1.set_title(f'{subject}', loc='left')
    if i ==0:
        ax1.set_ylabel('% Δ in trial time from BL', fontsize=fontsize)

    for DEG, indDEG, marker in zip([50,90],[ind50,ind90],['d', 'D']):
        mT = np.mean(dBEH[mode][i]['dist'][indDEG,:], axis=0)
        sT = stats.sem(dBEH[mode][i]['dist'][indDEG,:], axis=0)
        ax1.fill_between(np.arange(41), mT-sT, mT+sT, color=palette_ROT4[DEG], alpha=0.5)
    
        ANOVA_RES.append(dBEH[mode][i]['dist'][indDEG,0])

mode = 'shuffle'
open_file = open(os.path.join(root_path, 'pickles', f'dDates_dDegs_HDF_{mode}.pkl'), "rb")
dDates, dDegs, dDegs2 = pickle.load(open_file)
open_file.close()

for i in [0]:
    
    subj, subject = dSubject[i]
    ax1 = ax#[i]
    mT = np.mean(dBEH[mode][i]['time'], axis=0)
    sT = stats.sem(dBEH[mode][i]['time'], axis=0)
    

    #ax1.scatter(np.arange(41),mT, color='grey', marker='o', edgecolor='k')
    ax1.fill_between(np.arange(41), mT-sT, mT+sT, color='grey', alpha=0.5)

    #NOVA_INDS.append(np.ones(len(dDates[i]))*3)
    ANOVA_RES.append(dBEH[mode][i]['time'][:,0])


ax1.set_ylim([-5,120])
ax1.set_yticks(np.arange(0,121,20))
ax1.set_yticklabels(['BL', 20, 40, 60, 80, 100, 120])

plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
# plt.rcParams['font.family'] ='Times New Roman'
# plt.rcParams['mathtext.fontset'] = 'dejavusans'

for i in [0]:
    ax1.set_xticks([0,10-1,20-1,30-1,40-1])
    #ax[i].set_xticklabels([r'$1^{st}$', r'${10^{th}}$', '$20^{th}$', '$30^{th}$', '$40^{th}$'], rotation=0, fontstyle='italic')#, fontname='Arial') #\mathregular
    
    ax1.set_xlabel('perturbation trial set', fontsize=fontsize+2)
    
    ax1.axvline(0, color='k', ls='-.', zorder=0, lw=1)
    ax1.axhline(0, color='k', ls='-.', zorder=0, lw=1)
    
        
    ax1.set_ylim([-5,120])
    ax1.set_yticks(np.arange(0,121,20))
    ax1.set_yticklabels(['BL', 20, 40, 60, 80, 100, 120])
    
    # if i ==1:
    #     ax[i].set_yticklabels([])


#%%


"""

exponential fit

"""

  
        # xdata = np.arange(41)
        # ydata = mD
        # p0_a = mD[0] - np.min(mD)
        # p0_b = 0.5
        # p0_c = np.min(mD)
        # p0 = [p0_a, p0_b, p0_c]
        
        # popt, pcov = curve_fit(exponential_model, xdata, ydata, p0)
        # y_est = exponential_model(xdata,*popt)
        # R2 = compute_r_squared(ydata, y_est)
        # a,b,c = popt
        
        # print(f'{subject} | dist, {DEG} |  {a:.2f}*-e{b:.2f}t+{c:.2f} | R2={R2:.2f}')
        
        
    
        
        
        # # Calculate degrees of freedom
        # n = len(ydata)  # Number of data points
        # k = len(popt)   # Number of parameters
        # df = n - k
        
        # standard_errors = np.sqrt(np.diag(pcov))
        # t_values = popt / standard_errors
        # p_values = 2 * (1 - stats.t.cdf(np.abs(t_values), df))
        
        # print(f'{subject}| time | m={popt[0]:.2f} p={p_values[0]:.2f}')
        
        # ydata = mD
        # popt, pcov = curve_fit(linear_model, xdata, ydata)
        
        # # Calculate degrees of freedom
        # n = len(ydata)  # Number of data points
        # k = len(popt)   # Number of parameters
        # df = n - k
        
        # standard_errors = np.sqrt(np.diag(pcov))
        # t_values = popt / standard_errors
        # p_values = 2 * (1 - stats.t.cdf(np.abs(t_values), df))
        
        # print(f'{subject}| dist | m={popt[0]:.2f} p={p_values[0]:.2f}')
            


#%%


"""
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
############################################################################################################################################################
##############################################################################
"""



#%%


"""

Figure 6

"""


#%% 


#%%

"""

% sig

"""

# palette_ROT4 = {-90: blue, -50: yellow, 50: orange, 90: purple, 4: 'grey'}


# fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(7,7))


# for i in [0,1]:
    
#     subj, subject = dSubject[i]
    
#     ax[i].set_title(f'{subject}', loc='left')
    
#     assigned_ROT = np.array([np.round(np.median(dPD_each[i][date]['assigned_dPD'])) for date in dDates[i]])
#     sig_ROT = np.array([np.sum(dPD_each[i][date]['sig'])/len(dPD_each[i][date]['sig']) for date in dDates[i]])
#     sig_SHU = np.array([np.sum(dPD_each_shuffle[i][date]['sig'])/len(dPD_each_shuffle[i][date]['sig']) for date in dDates_shuffle[i]])


#     assigned_both = np.concatenate(( assigned_ROT, 4*np.ones(len(dDates_shuffle[i])) ))
#     sig_both = 100*np.concatenate(( sig_ROT, sig_SHU ))
    
    
#     sb.swarmplot(x=assigned_both, y=sig_both, ax=ax[i], order=[-90, -50, 50, 90, 4], palette=palette_ROT4)
#     #sb.swarmplot(x=assigned_ROT, y=sig_ROT, ax=ax[i], order=[-90, -50, 50, 90], palette=palette_ROT4)


#     ax[i].set_ylim([-5,100])
#     ax[i].set_ylabel('')
#     #ax[i].axhline(0, color='grey', ls='--', zorder=0)
    
#     ax[i].set_xticks([0,1,2,3,4])
#     ax[i].set_xticklabels(['-90°', '-50°', '+50°', '+90°', 'shuffle'], rotation=0) 
#     ax[i].set_xlabel('applied perturbation')
    
    
    
#     ind90_cw =  np.where(assigned_both == -90)[0] #-
#     ind50_cw =  np.where(assigned_both == -50)[0] #-
#     ind50_ccw = np.where(assigned_both == 50)[0]  #+
#     ind90_ccw = np.where(assigned_both == 90)[0]  #+
#     ind_shuffle = np.where(assigned_both == 4)[0]
    
    
#     v1 = sig_both[ind90_cw]
#     v2 = sig_both[ind50_cw]
#     v3 = sig_both[ind50_ccw]
#     v4 = sig_both[ind90_ccw]
#     v5 = sig_both[ind_shuffle]
    

    
#     ax[i].scatter([0,1,2,3,4], [np.mean(v1), np.mean(v2), np.mean(v3), np.mean(v4), np.mean(v5)], marker='s', color='w', edgecolor='k', linewidth=2.5,  s=70, zorder=10) #linewidth=2.5, s=75
#     (_, caps, _) = ax[i].errorbar([0,1,2,3,4], [np.mean(v1), np.mean(v2), np.mean(v3), np.mean(v4), np.mean(v5)], yerr=[np.std(v1), np.std(v2), np.std(v3), np.std(v4), np.std(v5)], color='k', lw=2.5, capsize=8, zorder=9, ls='')
#     for cap in caps:
#         cap.set_markeredgewidth(2.5)
        
    
    
#     dfANOVA = pd.DataFrame({'sig': sig_both,  'rot': assigned_both}) 
    
    
#     model = ols('sig ~ C(rot) ', data=dfANOVA).fit() 
#     result = sm.stats.anova_lm(model, type=1) 
      
#     print(result) 
    
#     tukey = pairwise_tukeyhsd(endog=dfANOVA['sig'],
#                               groups=dfANOVA['rot'],
#                               alpha=0.05)
    
#     print(tukey)
        
        

# ax[0].set_ylabel('% neurons with sig. measured ΔPD')


#%%

# """

# MD

# Average MD - Baseline
# Average MD - Perturbation

# Average Change in MD



# Session
# Then split by sig/not sig dPD


# """

# fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(8,8))


# for i in [0,1]:
    
#     subj, subject = dSubject[i]
    
#     ax[0][i].set_title(f'{subject}', loc='left')
    
#     assigned_ROT = np.array([np.round(np.median(dPD_each[i][date]['assigned_dPD'])) for date in dDates[i]])
    
#     sig_ROT = np.array([np.sum(dPD_each[i][date]['sig'])/len(dPD_each[i][date]['sig']) for date in dDates[i]])
#     sig_SHU = np.array([np.sum(dPD_each_shuffle[i][date]['sig'])/len(dPD_each_shuffle[i][date]['sig']) for date in dDates_shuffle[i]])
    
    
    
#     MD_BL_ROT = np.array([np.mean(dPD_each[i][date]['MD_BL']) for date in dDates[i]])
#     MD_PE_ROT = np.array([np.mean(dPD_each[i][date]['MD_PE']) for date in dDates[i]])

#     MD_BL_SHU = np.array([np.mean(dPD_each_shuffle[i][date]['MD_BL']) for date in dDates_shuffle[i]])
#     MD_PE_SHU = np.array([np.mean(dPD_each_shuffle[i][date]['MD_PE']) for date in dDates_shuffle[i]])

  


#     assigned_both = np.concatenate(( assigned_ROT, 4*np.ones(len(dDates_shuffle[i])) ))
#     sig_both = 100*np.concatenate(( sig_ROT, sig_SHU ))
#     MD_BL_BOTH = np.concatenate(( MD_BL_ROT, MD_BL_SHU ))
#     MD_PE_BOTH = np.concatenate(( MD_PE_ROT, MD_PE_SHU ))
    
    
#     sb.swarmplot(x=assigned_both, y=MD_BL_BOTH, ax=ax[0][i], order=[-90, -50, 50, 90, 4], palette=palette_ROT4)
#     sb.swarmplot(x=assigned_both, y=MD_PE_BOTH, ax=ax[1][i], order=[-90, -50, 50, 90, 4], palette=palette_ROT4)

#     # sb.swarmplot(x=assigned_ROT, y=MD_BL_ROT, ax=ax[0][i], order=[-90, -50, 50, 90], palette=palette_ROT4)
#     # sb.swarmplot(x=assigned_ROT, y=MD_PE_ROT, ax=ax[1][i], order=[-90, -50, 50, 90], palette=palette_ROT4)


#     ax[0][i].set_ylim([0,1])

#     ax[1][i].set_xticks([0,1,2,3,4])
#     ax[1][i].set_xticklabels(['-90°', '-50°', '+50°', '+90°', 'shuffle'], rotation=0) 
#     ax[1][i].set_xlabel('applied perturbation')
    

    
#     for j, MD_BLOCK in zip([0,1], [MD_BL_BOTH, MD_PE_BOTH]):
    
#         ind90_cw =  np.where(assigned_both == -90)[0] #-
#         ind50_cw =  np.where(assigned_both == -50)[0] #-
#         ind50_ccw = np.where(assigned_both == 50)[0]  #+
#         ind90_ccw = np.where(assigned_both == 90)[0]  #+
#         ind_shuffle = np.where(assigned_both == 4)[0]
        
        
#         v1 = MD_BLOCK[ind90_cw]
#         v2 = MD_BLOCK[ind50_cw]
#         v3 = MD_BLOCK[ind50_ccw]
#         v4 = MD_BLOCK[ind90_ccw]
#         v5 = MD_BLOCK[ind_shuffle]

        
#         ax[j][i].scatter([0,1,2,3,4], [np.mean(v1), np.mean(v2), np.mean(v3), np.mean(v4), np.mean(v5)], marker='s', color='w', edgecolor='k', linewidth=2.5,  s=70, zorder=10) #linewidth=2.5, s=75
#         (_, caps, _) = ax[j][i].errorbar([0,1,2,3,4], [np.mean(v1), np.mean(v2), np.mean(v3), np.mean(v4),np.mean(v5)], yerr=[np.std(v1), np.std(v2), np.std(v3), np.std(v4), np.std(v5)], color='k', lw=2.5, capsize=8, zorder=9, ls='')
#         for cap in caps:
#             cap.set_markeredgewidth(2.5)
        

# #ax[0].set_ylabel('% neurons with significant measured ΔPD')

# ax[0][0].set_ylabel('average MD (baseline)')
# ax[1][0].set_ylabel('average MD (perturbation)')



"""

MD - FOR N.S. dPD

Average MD - Baseline
Average MD - Perturbation


"""

fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(8,8))


for i in [0,1]:
    
    subj, subject = dSubject[i]
    
    ax[0][i].set_title(f'{subject}', loc='left')
    ax[0][i].text(-0.25,0.95,'neurons with n.s. measured ΔPD only', fontstyle='italic')
    ax[1][i].text(-0.25,0.95,'neurons with n.s. measured ΔPD only', fontstyle='italic')
    
    assigned_ROT = np.abs(np.array([np.round(np.median(dPD_each[i][date]['assigned_dPD'])) for date in dDates[i]]))
    
    MD_BL_ROT = []
    MD_PE_ROT = []
    for date in dDates[i]:
        
        sig_ = dPD_each[i][date]['sig']
        
        sigF = np.where(sig_ == 0)[0]
        
        MD_BL_ = dPD_each[i][date]['MD_BL'][sigF]
        MD_PE_ = dPD_each[i][date]['MD_PE'][sigF]
        
        MD_BL_ROT.append(np.mean(MD_BL_))
        MD_PE_ROT.append(np.mean(MD_PE_))
        
    
    MD_BL_SHU = []
    MD_PE_SHU = []
    for date in dDates_shuffle[i]:
        
        sig_ = dPD_each_shuffle[i][date]['sig']
        
        sigF = np.where(sig_ == 0)[0]
        
        MD_BL_ = dPD_each_shuffle[i][date]['MD_BL'][sigF]
        MD_PE_ = dPD_each_shuffle[i][date]['MD_PE'][sigF]
        
        MD_BL_SHU.append(np.mean(MD_BL_))
        MD_PE_SHU.append(np.mean(MD_PE_))



    assigned_both = np.concatenate(( assigned_ROT, 4*np.ones(len(dDates_shuffle[i])) ))

    MD_BL_BOTH = np.concatenate(( MD_BL_ROT, MD_BL_SHU ))
    MD_PE_BOTH = np.concatenate(( MD_PE_ROT, MD_PE_SHU ))
    



    

    
    sb.swarmplot(x=assigned_both, y=MD_BL_BOTH, ax=ax[0][i], order=[ 50, 90, 4], palette=palette_ROT4)
    sb.swarmplot(x=assigned_both, y=MD_PE_BOTH, ax=ax[1][i], order=[50, 90, 4], palette=palette_ROT4)


    for j, MD_BLOCK in zip([0,1], [MD_BL_BOTH, MD_PE_BOTH]):
        
        # ind90_cw =  np.where(assigned_both == -90)[0] #-
        # ind50_cw =  np.where(assigned_both == -50)[0] #-
        ind50_ccw = np.where(assigned_both == 50)[0]  #+
        ind90_ccw = np.where(assigned_both == 90)[0]  #+
        ind_shuffle = np.where(assigned_both== 4)[0]
        
        
        #v1 = MD_BLOCK[ind90_cw]
        # v2 = MD_BLOCK[ind50_cw]
        v3 = MD_BLOCK[ind50_ccw]
        v4 = MD_BLOCK[ind90_ccw]
        v5 = MD_BLOCK[ind_shuffle]


        ax[j][i].scatter([0,1,2], [ np.mean(v3), np.mean(v4), np.mean(v5)], marker='s', color='w', edgecolor='k', linewidth=2.5,  s=70, zorder=10) #linewidth=2.5, s=75
        (_, caps, _) = ax[j][i].errorbar([0,1,2], [np.mean(v3), np.mean(v4),np.mean(v5)], yerr=[ np.std(v3), np.std(v4), np.std(v5)], color='k', lw=2.5, capsize=8, zorder=9, ls='')
        for cap in caps:
            cap.set_markeredgewidth(2.5)
            
    ax[0][i].set_ylim([0,1])
    # ax[1][i].set_xticks([0,1,2,3,4])
    # ax[1][i].set_xticklabels(['-90°', '-50°', '+50°', '+90°', 'shuffle'], rotation=0) 
    # ax[1][i].set_xlabel('applied perturbation')
    ax[1][i].set_xticks([0,1,2])
    ax[1][i].set_xticklabels(['rotation,\neasy', 'rotation,\nhard', 'shuffle'], rotation=0) 
    ax[1][i].set_xlabel('applied perturbation')
    
        
    

        

#ax[0].set_ylabel('% neurons with significant measured ΔPD')

ax[0][0].set_ylabel('average MD (baseline)')
ax[1][0].set_ylabel('average MD (perturbation)')












#%%

"""
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
############################################################################################################################################################
##############################################################################
"""


#Figure 7 - Rotation








# scBL_100ms = np.mean(dSC['BL']/(n_window*0.1), axis=0)*0.1
# KY_ = np.sqrt((xBL*scBL_100ms)**2 + (yBL*scBL_100ms)**2)
# KY.append(100*(KY_/np.sum(KY_))) 

# print(np.sum(KY_))
# print(np.linalg.norm(KY_))

# x = xBL*scBL_100ms
# y = yBL*scBL_100ms

# temp = np.sqrt(x**2 + y**2)

# plt.scatter(KY_, temp)


# for xi, yi in zip(x[:1],y[:1]):
    
#     plt.plot([0,xi],[0,yi], color='k')
    
#     plt.plot([0,xi],[0,0], color='grey', ls='--')
#     plt.plot([xi,xi],[0,yi], color='grey', ls='--')
    
#     plt.scatter(0, np.sqrt( xi**2 + yi**2) , color='blue')



#%%

# for i in [0,1]:
    
#     dfI = dRES[i]

#     tot50 = dfI.loc[ (dfI['a_dPD']==50) | (dfI['a_dPD']==-50) ]
#     sv = tot50.loc[tot50['sig']==1, 'svBL']
#     ky = tot50.loc[tot50['sig']==1, 'KY']
#     r,p,star = compute_corr(sv,ky)
#     print(f'{i}, {len(sv)}, [50] | r={r:.2f}, p={p:.2e}, {star}')
    
    
#     tot90 = dfI.loc[ (dfI['a_dPD']==90) | (dfI['a_dPD']==-90) ]
#     sv = tot90.loc[tot90['sig']==1, 'svBL']
#     ky = tot90.loc[tot90['sig']==1, 'KY']
#     r,p,star = compute_corr(sv,ky)
#     print(f'{i}, {len(sv)}, [90] | r={r:.2f}, p={p:.2e}, {star}')
    
    
    
    # #print(len(tot50), len(tot90))
    
    # # print('EASY', i, len(tot50.loc[tot50['sig']==1]), len(tot50))
    # # print('HARD', i, len(tot90.loc[tot90['sig']==1]), len(tot90))
    
#%%


"""


supplemental

"""


#%%

"""

##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################
"""



#%%

"""

additional plots not presented in any figures...

"""



  
#%%

"""

[not presented]

ROTATION
    Session Averages [dPD_16, dPD_median, dPD_84]

"""

# PD_KEY = 'dPD_84'

# fontsize = 12
# mpl.rcParams["font.size"] = fontsize

# fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(6,6))

# for i in [0,1]:
    
#     subj, subject = dSubject[i]
    
#     ax[i].set_title(f'{subject}', loc='left')

#     assigned = np.array([np.round(np.median(dPD_each[i][date]['assigned_dPD'])) for date in dDates[i]])


#     for d in range(len(dDates[i])):
        
#         ax[i].plot([d,d],[dPD_all[i]['dPD_16'][d], dPD_all[i]['dPD_84'][d]], color=palette_ROT4[assigned[d]])
    

#     ax[i].axhline(0, color='grey', ls='--', zorder=0)
#     ax[i].set_ylim([-100,100])

#     ax[i].axhline(-90, color=blue, lw=2, zorder=0)#, ls='--', zorder=0)
#     ax[i].axhline(-50, color=yellow, lw=2, zorder=0)#, ls='--', zorder=0)
#     ax[i].axhline(50, color=orange, lw=2, zorder=0)#, ls='--', zorder=0)
#     ax[i].axhline(90, color=purple, lw=2, zorder=0)#, ls='--', zorder=0)
    
#     # ind90_cw =  np.where(assigned == -90)[0] #-
#     # ind50_cw =  np.where(assigned == -50)[0] #-
#     # ind50_ccw = np.where(assigned == 50)[0]  #+
#     # ind90_ccw = np.where(assigned == 90)[0]  #+
    
    
#     # v1 = dPD_all[i][PD_KEY][ind90_cw]
#     # v2 = dPD_all[i][PD_KEY][ind50_cw]
#     # v3 = dPD_all[i][PD_KEY][ind50_ccw]
#     # v4 = dPD_all[i][PD_KEY][ind90_ccw]
    
#     # ax[i].scatter([0,1,2,3], [np.mean(v1), np.mean(v2), np.mean(v3), np.mean(v4)], marker='s', color='w', edgecolor='k', linewidth=1.5,  s=50, zorder=10) #linewidth=2.5, s=75
#     # # (_, caps, _) = ax[i].errorbar([0,1,2,3], [np.mean(v1), np.mean(v2), np.mean(v3), np.mean(v4)], yerr=[np.std(v1), np.std(v2), np.std(v3), np.std(v4)], color='k', lw=2.5, capsize=8, zorder=9, ls='')
#     # # for cap in caps:
#     # #     cap.set_markeredgewidth(2.5)
    
#     ax[i].set_xlabel('session number (arbitrary)')
        
# ax[0].set_ylabel('measured ΔPD [16th-84th percentile] (°)')






    
#%%

"""

[not presented ]
    example ROTATION session: plotting MD vs dPD

"""

# dPD_each_ = dPD_each

# palette_ROT4 = {-90: blue, -50: yellow, 50: orange, 90: purple, 4: 'grey'}


# fontsize = 14
# mpl.rcParams["font.size"] = fontsize

# fig, ax = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=False, figsize=(10,15))

# for i in [0,1]:
    
#     subj, subject = dSubject[i]
    
#     #assigned = np.array([np.round(np.median(dPD_each[i][date]['assigned_dPD'])) for date in dDates[i]])


#     ax[0][i].set_title(f'{subject}', loc='left')
    
    
#     d = 10
#     date = dDates[i][d]
    
    

#     sig = dPD_each_[i][date]['sig']
    
#     dPD_lo  = dPD_each_[i][date]['dPD_lo']
#     dPD_med = dPD_each_[i][date]['dPD_median']
#     dPD_hi  = dPD_each_[i][date]['dPD_hi']
    
#     MD_BL = dPD_each_[i][date]['MD_BL']
#     MD_PE = dPD_each_[i][date]['MD_PE']
    
#     dfPLOT = pd.DataFrame({'sig': sig, 
#                            'dPD_med': dPD_med, 'dPD_lo': dPD_lo, 'dPD_hi': dPD_hi,
#                            'MD_BL': MD_BL, 'MD_PE': MD_PE}).sort_values(by=['sig', 'dPD_med']).reset_index(drop=True)

    
    
#     axis1 = ax[0][i]
#     axis2 = ax[1][i]
#     axis3 = ax[2][i]
#     for l, MD1, MD2, sig_j, PD1, PD2 in zip(np.arange(len(dfPLOT)), dfPLOT['MD_BL'], dfPLOT['MD_PE'], dfPLOT['sig'], dfPLOT['dPD_lo'], dfPLOT['dPD_hi']):
        
        
#         if sig_j == 0:
#             color = 'white'
#             marker='o'
#         elif sig_j == 1:
#             color = 'k'
#             marker='s'
        
#         axis1.scatter(MD1, MD2, marker=marker, color=color, edgecolor='k', alpha=0.6)
        
#         axis2.scatter(MD1, PD2 - PD1, marker=marker, color=color, edgecolor='k', alpha=0.6)
#         axis3.scatter(MD2, PD2 - PD1, marker=marker, color=color, edgecolor='k', alpha=0.6)

#     axis1.set_xlabel('MD (baseline)')
#     axis1.set_ylabel('MD (perturbation)')

#     axis1.set_ylim([0,2]) 
#     axis1.set_yticks([0,0.5,1,1.5,2])
#     axis1.plot([0,2],[0,2], color='b', zorder=0)
    
    
#     axis2.set_ylabel('range of 95% CI of measured ΔPD')
#     axis2.set_xlabel('MD (baseline)')
#     axis2.set_ylim([0,360])
#     axis2.set_yticks(np.arange(0,361,45))
    
#     axis3.set_ylabel('range of 95% CI of measured ΔPD')
#     axis3.set_xlabel('MD (perturbation)')
#     axis3.set_ylim([0,360])
#     axis3.set_yticks(np.arange(0,361,45))
    
    
#     for axis in [axis1, axis2, axis3]:
#         axis.set_xlim([-0.1,2.1]) 
#         axis.set_xticks([0,0.5,1,1.5,2])






