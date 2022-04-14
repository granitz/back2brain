# Content
# 1. define groups
# Hypothesis I:
# 2. Run RPN (pain sensitivity)
# 3. analyse framewise displacement
# 4. analyse RPN
# Hypothesis III:
# 5. remove nans from fmriprep confounds
# 6. Extract timeseries and clean data
# 7. Scrubbing.
# 8. Report number/percentage of scrubbed volumes per group and if any excluded
# 10. Statistics section ( permutation test )
# 11. Exploratory analysis ( network theory on random graphs )
# ----------------------------#

###
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
plt.ion()
import seaborn as sns


# 1.
# groups
cases_list = []
controls_list = []
cases_list.sort()
controls_list.sort()

# 2.
### Pain sensitivity
docker run -it --rm -v /data/grakas/kirurgi/nifti/:/data:ro -v /data/grakas/kirurgi/nifti/derivatives:/out \
tspisak/rpn-signature:latest /data /out/RPN participant --mem_gb 10 --nthreads 7 --2mm

# 3.
# Import framewise displacement
FD = pd.read_csv('file:///data/grakas/kirurgi/nifti/derivatives/RPN/motion_summary.csv',',').drop('Unnamed: 0',1)
FD['group'] = RPN['group'] # please see below (under RPN stuff)

# cases
A = FD[FD.group==1].meanFD.values
# controls
B = FD[FD.group==0].meanFD.values

# Normality check h0: sample comes from a normal distribution
statisticA, pvalA = scipy.stats.normaltest(A) # log for normality
statisticB, pvalB = scipy.stats.normaltest(B)

# Equal variances. h0: all input samples are from populations with equal variances.
statistic, p = scipy.stats.levene(np.log(A),np.log(B))

# ttest
T,p = scipy.stats.ttest_ind(np.log(A), np.log(B))

# Run correlation between FD and covariates (not part of prereg)

# 4.
# Compare groups for RPN
RPN = pd.read_csv('/data/grakas/kirurgi/nifti/derivatives/RPN/RPNresults.csv').drop('Unnamed: 0',1)

tmplist = []
for x in RPN.in_file:
    sub = x.split('/')[2] # prints sub-xxxx
    if sub in cases_list:
        tmplist.append(1)
    elif sub in controls_list:
        tmplist.append(0)

RPN['group'] = tmplist

# cases
A = RPN[RPN.group==1].RPN.values
# controls
B = RPN[RPN.group==0].RPN.values

# Normality check h0: sample comes from a normal distribution
statisticA, pvalA = scipy.stats.normaltest(A)
statisticB, pvalB = scipy.stats.normaltest(B)

# Equal variances. h0: all input samples are from populations with equal variances.
statistic, p = scipy.stats.levene(A,B, center='median') # median for skewed data

# ttest
T,p = scipy.stats.ttest_ind(A, B, equal_var=False)

# Run correlation between RPN and covariates
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests

df = 'back2brain_covariates_including_RPN_and_meanFD.csv'
df.drop(['MR_RPN','meanFD'],1,inplace=True)
df = df.iloc[:,0:16] # Index -> ['RPN', 'Group', 'VAS_Back_Correct', 'VAS_Leg_Correct','ODI_Score_Exam', 'EQ_Index_Exam', 'PCS_Exam', 'MCS_Exam', 'L4_L5_Pf', 'L5_S1_Pf', 'MCPA_L4_L5', 'MCPA_L5_S1', 'TEP_L4_L5', 'TEP_L5_S1','mean_trunc_angle', 'mean_headangle']

# cases
cases = df[df.Group==1]
# control
control = df[df.Group==2]

X = controls['RPN']
rvals = []
pvals = []
levene = []
for x in range(2,16):
    Y = controls.iloc[:,x]
    cond = Y.isna()
    # ignore nans
    X = X[~cond]
    Y = Y[~cond]

    statistic, p = scipy.stats.levene(X,Y)
    levene.append(p)

    if p>0.05:
        r, p = scipy.stats.spearmanr(X,Y)
    else:
        r,p = pearsonr(X,Y)
    rvals.append(r)
    pvals.append(p)

mtest = multipletests(pvals,method='fdr_bh')

# 5.
# remove nans from fmriprep confounds
import os
PATH = '/data/grakas/kirurgi/nifti/derivatives/fmriprep/'
subs = os.listdir(PATH)
subs = [x for x in subs if 'html' not in x]
subs = [x for x in subs if x!='logs']
subs = [x for x in subs if 'dataset' not in x]
nsubs = len(subs)

for ii, sub in enumerate(subs):

    data = '/data/grakas/kirurgi/nifti/derivatives/fmriprep/{}/func/{}_task-rest_desc-confounds_regressors.tsv'.format(sub,sub)

    df = pd.read_csv(data, '\t', index_col=0)
    df.fillna(0, inplace=True)
    df.to_csv('/data/grakas/kirurgi/nifti/derivatives/fmriprep/{}/func/{}_task-rest_desc-confounds_regressors.tsv'.format(sub,sub), sep='\t')

############################################
### 6. Extract timeseries and clean data ###
############################################
# either using teneto, nideconv or directly from denoised data in CONN

#### use teneto fmriprep timeseries and clean data
# from teneto import TenetoBIDS
# bids = '/data/grakas/kirurgi/nifti/'
# tnet = TenetoBIDS(bids,selected_pipeline='fMRIPrep',exist_ok=True)
#
# nilearn_params = {'standardize':True,'low_pass':0.09,'high_pass':0.008,'t_r':2.205}
# parcellation_params = {'atlas':'Schaefer2018',
#                         'atlas_desc':'400Parcels7Networks',
#                         'resolution':1,
#                         'parc_params': nilearn_params}
#
# tnet.run('make_parcellation',input_params=parcellation_params)

#
# #### or get data from nideconv and clean
# from nilearn import datasets
# atlas = datasets.fetch_atlas_schaefer_2018
# from nideconv.utils.roi import get_fmriprep_timeseries
# ts = get_fmriprep_timeseries(fmriprep_data,
#                                  source_data,
#                                  atlas)


#### or extract denoised data from from conn ##############################################
import numpy as np
import pandas as pd
import os
import scipy.io as sio
import matplotlib.pyplot as plt
plt.ion()
import seaborn as sns

PATH = '/data/grakas/Program/connprojects/conn_project_back2brain/results/preprocessing/'
files = os.listdir(PATH)
files = [x for x in files if 'Condition001' in x ]
files.sort()

subs = [x for x in os.listdir('/data/grakas/kirurgi/nifti/') if 'sub' in x]
subs.sort()

nnodes_tiun, nnodes_schaefer = 32, 400
nnodes = nnodes_tiun + nnodes_schaefer
nstart, nend = 3, 435 # nodes 0 to 2 are other variables e.g. csf. Nodes beyond 435 are confounds regressors.
nsubs = len(files)

net = np.zeros((nsubs, nnodes_tiun+nnodes_schaefer,220))

savepath = '/data/grakas/kirurgi/nifti/derivatives/conn_denoised_ts/'
if ~os.path.exists(savepath):
    os.mkdir(savepath)

files = [x for x in files if 'ROI' in x]

for ii, cond in enumerate(files):
    datapath = PATH + cond

    if not os.path.exists(savepath + subs[ii] + '/'):
        os.mkdir(savepath + subs[ii])
        os.mkdir(savepath + subs[ii] + '/func/')

    xyz = sio.loadmat(datapath)['xyz'][0,nstart:nend]
    xyz2 = [np.squeeze(x) for x in xyz]
    df2 = pd.DataFrame(xyz2, columns=['x','y','z'])
    names = sio.loadmat(datapath)['names'][0,nstart:nend]
    names = np.array([x[0].split('.')[1] for x in names])
    data = sio.loadmat(datapath)['data'][0,nstart:nend]

    for jj in range(nnodes):
        net[ii, jj, :] = np.squeeze(data[jj])

    df = pd.DataFrame(net[ii,:,:])
    df.index = names
    if os.path.exists(savepath + subs[ii] + '/func/' + subs[ii] + '_run-1_task-rest_desc-conn-name-' + str(ii+1) + '_timeseries.tsv'):
        os.remove(savepath + subs[ii] + '/func/' + subs[ii] + '_run-1_task-rest_desc-conn-name-' + str(ii+1) + '_timeseries.tsv')
    df.to_csv(savepath + subs[ii] + '/func/' + subs[ii] + '_run-1_task-rest_desc-conn-name-' + str(ii+1) + '_timeseries.tsv', '\t')
    # df.to_csv(savepath +  subs[ii] + '/func/conn-denoised_conn-name-' + str(ii+1) + '.tsv','\t') # both files and subs are sorted

# 7.
# Scrubbing
### import the conn-denoised data with teneto and scrubb FD >0.5 more than 25% ################
from scipy.interpolate import interp1d
import json

confound_name = 'framewise_displacement'
replace_with = 'cubicspline'
relex = np.greater
crit = 0.5
tol=0.25

loaddir = '/data/grakas/kirurgi/nifti/derivatives/conn_denoised_ts/'
savedir = '/data/grakas/kirurgi/nifti/derivatives/teneto-censor-timepoints/'

if not os.path.exists(savedir):
    os.mkdir(savedir)

metadata = {}
for ii, cond in enumerate(files):
    datapath = PATH + cond

    if not os.path.exists(savedir + subs[ii]):
        os.mkdir(savedir + subs[ii])
        os.mkdir(savedir + subs[ii] + '/func/')

    # load data
    timeseries = pd.read_csv(loaddir + subs[ii] + '/func/' + subs[ii] + '_run-1_task-rest_desc-conn-name-' + str(ii+1) + '_timeseries.tsv', '\t',index_col=0)

    # load confounds
    confounds = pd.read_csv('/data/grakas/kirurgi/nifti/derivatives/fmriprep/' + subs[ii] + '/func/' + subs[ii] +  '_task-rest_desc-confounds_regressors.tsv', sep = '\t', header = 0)
    Z = timeseries.copy()
    ### clean data ###############################################################################################
    ci = confounds[confound_name]
    bad_timepoints = list(ci[relex(ci, crit)].index)
    bad_timepoints = list(map(str, bad_timepoints))
    timeseries[bad_timepoints] = np.nan
    if replace_with == 'cubicspline' and len(bad_timepoints) > 0:
        good_timepoints = sorted(
            np.array(list(map(int, set(timeseries.columns).difference(bad_timepoints)))))
        bad_timepoints = np.array(list(map(int, bad_timepoints)))
        ts_index = timeseries.index
        timeseries = timeseries.values
        bt_interp = bad_timepoints[bad_timepoints > np.min(good_timepoints)]
        for n in range(timeseries.shape[0]):
            interp = interp1d(
                good_timepoints, timeseries[n, good_timepoints], kind='cubic')
            timeseries[n, bt_interp] = interp(bt_interp)
        timeseries = pd.DataFrame(timeseries, index=ts_index)
        bad_timepoints = list(map(str, bad_timepoints))
    ###############################################################################################

    Z = Z.values - timeseries.values
    print(Z.any())
    # save
    if os.path.exists(savedir + subs[ii] + '/func/' + subs[ii] + '_run-1_task-rest_desc-conn-name-' + str(ii+1) + '_timeseries.tsv'):
        os.remove(savedir + subs[ii] + '/func/' + subs[ii] + '_run-1_task-rest_desc-conn-name-' + str(ii+1) + '_timeseries.tsv')

    timeseries.to_csv(savedir + subs[ii] + '/func/' + subs[ii] + '_run-1_task-rest_desc-conn-name-' + str(ii+1) + '_timeseries.tsv', '\t')
    # df.to_csv(savepath +  subs[ii] + '/func/conn-denoised_conn-name-' + str(ii+1) + '.tsv','\t') # both files and subs are sorted
    metadata[subs[ii] + ' num of censored volumes'] = len(bad_timepoints)

    if len(bad_timepoints)>(timeseries.shape[1]*0.25):
        print('Number of bad timepoints for subject ' + subs[ii] + '=' + str(len(bad_timepoints)))

if os.path.exists('/data/grakas/kirurgi/nifti/derivatives/teneto-censor-timepoints/metadata.json'):
    os.remove('/data/grakas/kirurgi/nifti/derivatives/teneto-censor-timepoints/metadata.json')

with open('/data/grakas/kirurgi/nifti/derivatives/teneto-censor-timepoints/metadata.json','w') as f:
    json.dump(metadata, f)

# 8.
# Report number/percentage of scrubbed volumes per group and if any excluded.

def reportNumScrubbedFrames(cases_l,controls_l):
    """Takes list of names for cases and controls and prints
        the mean and standard deviation of num scrubbed volumes
        and returns list per group of number of scrubbed vols"""

    metadata = '/data/grakas/kirurgi/nifti/derivatives/teneto-censor-timepoints/metadata.json'
    f = json.load(open(metadata,'rb'))

    num_vol_scrubbed_cases = []
    num_vol_scrubbed_controls = []

    for sub in f.keys():
        sub_name = sub.split(' ')[0]
        if sub_name in cases_l:
            num_vol_scrubbed_cases.append(f[sub])
        if sub_name in controls_l:
            num_vol_scrubbed_controls.append(f[sub])

    mean_cases = np.mean(num_vol_scrubbed_cases)
    std_cases = np.std(num_vol_scrubbed_cases)
    mean_controls = np.mean(num_vol_scrubbed_controls)
    std_controls = np.std(num_vol_scrubbed_controls)
    print('Cases: Mean = ' + str(mean_cases) + ', standard deviation = ' + str(std_cases))
    print('Controls: Mean = ' + str(mean_controls) + ', standard deviation = ' + str(std_controls))

    return num_vol_scrubbed_cases, num_vol_scrubbed_controls


# visualize CONN roi2roi Allsubjects, unthresholded & TFCE
def visualize_roi2roi_tfce(cases_l,controls_l):
    # bids named participants (sorted). Should match the order in CONN since it too orders.
    subs = [x for x in os.listdir('/data/grakas/kirurgi/nifti/') if 'sub' in x]
    subs.sort()

    # load nodes as sorted by CONN. Transforms from strings like 'schaefer.030_blabla' to 30_blabla.
    f = json.load(open('/data/grakas/Program/connprojects/conn_project_back2brain/results/secondlevel/SBC_02_ROI2ROI/AllSubjects/rest/exported_data.json','rb'))
    tmp = f['names']
    conn_sorted_extract_int = [node_name.split('.')[1] for node_name in tmp]
    del tmp

    # determing the vector to sort matrices
    sorting_vec = []
    for ii, x in enumerate(names):
        if x in conn_sorted_extract_int:
            idx = np.where(np.array(conn_sorted_extract_int)==x)[0]
            sorting_vec.append(idx[0])

    # load correct nifti unth
    unth_data = np.squeeze(nib.load('/data/grakas/Program/connprojects/conn_project_back2brain/results/secondlevel/SBC_02_ROI2ROI/AllSubjects/rest/exported_data.nii').get_fdata())
    group_avg = np.mean(unth_data,axis=-1)

    # sort_group_avg according to schaefer and tiuan instead of CONN's hierarchical clustering
    group_avg_sorted = group_avg[sorting_vec,:][:,sorting_vec]

    # collect cases and controls into individuals buckets
    cases_data = []
    controls_data = []
    for sub_idx, sub in enumerate(subs):
        if sub in cases_l:
            corr = unth_data[:,:,sub_idx]
            # sort
            corr = corr[sorting_vec,:][:,sorting_vec]
            cases_data.append(corr)
        if sub in controls_l:
            corr = unth_data[:,:,sub_idx]
            # sort
            corr = corr[sorting_vec,:][:,sorting_vec]
            controls_data.append(unth_data[:,:,sub_idx])

    cases_data = np.array(cases_data)
    controls_data = np.array(controls_data)
    mean_cases = np.mean(cases_data,axis=0)
    mean_controls = np.mean(controls_data,axis=0)

    positive_cases = mean_cases.copy()
    positive_cases[positive_cases<0] = 0

    positive_controls = mean_controls.copy()
    positive_controls[positive_controls<0] = 0

    negative_cases = mean_cases.copy()
    negative_cases[negative_cases>0] = 0

    negative_controls = mean_controls.copy()
    negative_controls[negative_controls>0] = 0

    # positive diff:
    diffp = positive_cases - positive_controls
    # negative diff:
    diffn = abs(negative_cases) - abs(negative_controls )

    # load correct nifti tfce
    tfce = nib.load('/data/grakas/Program/connprojects/conn_project_back2brain/results/secondlevel/SBC_02_ROI2ROI/AllSubjects/rest/results.nii').get_fdata()
    tfce = tfce[sorting_vec,:][:,sorting_vec]

    # NOTE! instead of plotting tfce results, use the group averaged results and mask with clusters as determined with tfce
    results_mask = nib.load('/data/grakas/Program/connprojects/conn_project_back2brain/results/secondlevel/SBC_02_ROI2ROI/AllSubjects/rest/results.mask.nii').get_fdata()
    results_mask = results_mask[sorting_vec,:][:,sorting_vec]
    results_mask = results_mask.astype(int)
    masked_group_avg = group_avg_sorted.copy()
    masked_group_avg[results_mask==0] = 999
    tfce = masked_group_avg.copy()

    # plot whole matrix. TFCE.

    # For colors, see below, before computing cartographic profile.

    # leave 10 rows and cols empty
    group_mat = np.zeros((10+group_avg_sorted.shape[0],10+group_avg_sorted.shape[0]))
    tfce_mat = np.zeros((10+tfce.shape[0],10+tfce.shape[0]))

    # get lower triangle of
    mask_group_avg = np.triu(np.ones_like(group_avg_sorted, dtype=bool))
    mask_tfce = np.triu(np.ones_like(tfce, dtype=bool))

    # fill with values
    tfce = tfce.T
    group_avg_sorted[mask_group_avg] = 999
    tfce[~mask_tfce] = 999
    group_mat[:-10,10:] = group_avg_sorted
    tfce_mat[:-10,10:] = tfce

    # Used for plotting patches along the x axis and y-axis (transposed)
    mask = np.zeros((10,432+10))
    mask[mask==0]=999
    group_mat[432:,:]= mask
    group_mat[:,:10] = mask.T
    tfce_mat[432:,:] = mask
    tfce_mat[:,:10]= mask.T


    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    fig,axes = plt.subplots(1,2,figsize=(12,6))

    sns.heatmap(group_mat,cmap=cmap,mask=group_mat==999,square=True, center=0,cbar_kws={'shrink':.3,'use_gridspec':False,'location':'top'},ax=axes[0])
    sns.heatmap(tfce_mat,cmap=cmap,mask=tfce_mat==999,square=True,center=0,cbar_kws={'shrink':.3,'use_gridspec':False,'location':'top'},ax=axes[1])

    axes[0].collections[0].colorbar.set_label('Mean FC $(z)$',fontsize='large')
    axes[0].collections[0].colorbar.set_ticks([-0.6,0,1.2])
    #ax.collections[0].colorbar.set_ticklabels([r'Int',r'Seg'])
    # ax.collections[0].colorbar.ax.tick_params(labelsize=14)
    # ax.collections[0].colorbar.ax.get_xaxis().set_label_coords(0.5,2)
    axes[1].collections[0].colorbar.set_label('Mean FC $(z)$ (masked)',fontsize='large')
    axes[1].collections[0].colorbar.set_ticks([-0.6,0,1.2])

    axes[0].set_yticks([])
    axes[0].set_yticks([])
    axes[0].set_xticks([])
    axes[1].set_yticks([])
    axes[1].set_yticks([])
    axes[1].set_xticks([])

    # x axis
    y=432
    height=9
    width=1
    alpha = [0.4,0.5,0.6,0.7,0.8]
    for ii in [0,1]:
        for step in range(432):
            axes[ii].add_patch(matplotlib.patches.Rectangle((10+step,y), width,height, color=colors[step],alpha=0.25))

    # y axis
    y=0
    height=1
    width=9
    # only net
    for ii in [0,1]:
        for step in range(432):
            axes[ii].add_patch(matplotlib.patches.Rectangle((y,step), width, height, color=colors[step],alpha=0.25))

    return ax, cases_data, controls_data



# 9.
# Median/max PC/Z
### pearson/partial correlation ########################################################################
# load data per subject.
# threshold according to pre-registration
# compute participation coef and module degree zscore measure

# ipython 7.19.0, python 3.8.5.
import pingouin # version '0.3.8'
import bct # version '0.5.0'
import community # version '0.14'
import networkx as nx # version '2.5'
import pandas as pd # version '1.1.3'
import numpy as np # version '1.19.2'
import matplotlib.pyplot as plt # '3.3.2'
import seaborn as sns # '0.11.0'
import scipy.io as sio
import os
import json
import nibabel as nib
plt.ion()

loaddir = '/data/grakas/kirurgi/nifti/derivatives/conn_denoised_ts/'
savedirpartial = '/data/grakas/kirurgi/analysis/cartographic_profile/partial/'
savedirpearson = '/data/grakas/kirurgi/analysis/cartographic_profile/pearsonr/'
savedirconn = '/data/grakas/kirurgi/analysis/cartographic_profile/conn/'
savedirconn = '/data/grakas/kirurgi/analysis/cartographic_profile/conn_nodal_level/'

savedirs = [savedirpearson,savedirpartial,savedirconn]

subs = [x for x in os.listdir('/data/grakas/kirurgi/nifti/') if 'sub' in x]
subs.sort()
nsubs = len(subs)
nnodes = 32+400 # Schaefer + Tian subcortial

# collect all participants
net = np.zeros((nsubs, nnodes,nnodes))
netunthresholded = np.zeros((nsubs, nnodes,nnodes))

# collect into groups
cases = np.zeros((len(cases_list),nnodes,nnodes))
controls = np.zeros((len(controls_list),nnodes,nnodes))

cases_unthresholded = np.zeros((len(cases_list),nnodes,nnodes))
controls_unthresholded = np.zeros((len(controls_list),nnodes,nnodes))

# collect participation_coef, module_degree_zscore. Don't need to save communities.
participation_coef = np.zeros((nsubs, nnodes))
module_degree_zscore = np.zeros((nsubs, nnodes))
pc_cases = np.zeros((len(cases_list),nnodes))
z_cases = np.zeros((len(cases_list),nnodes))
pc_controls = np.zeros((len(controls_list),nnodes))
z_controls = np.zeros((len(controls_list),nnodes))

### max median pc/z
maxPC = np.zeros((nsubs, 8))
maxZ = np.zeros((nsubs,8))

medianPC = np.zeros((nsubs,8))
medianZ = np.zeros((nsubs,8))

# groups
maxPC_cases = np.zeros((len(cases_list), 8))
maxZ_cases = np.zeros((len(cases_list),8))
medianPC_cases = np.zeros((len(cases_list),8))
medianZ_cases = np.zeros((len(cases_list),8))

maxPC_controls = np.zeros((len(controls_list), 8))
maxZ_controls = np.zeros((len(controls_list),8))
medianPC_controls = np.zeros((len(controls_list),8))
medianZ_controls = np.zeros((len(controls_list),8))


maxPCidx = {}
maxZidx = {}

medianPCidx = {}
medianZidx = {}

#### for importing xyz and community_names from conn-toolbox
nnodes_tiun, nnodes_schaefer = 32, 400
nstart, nend = 3, 435 # conn toolbox returns node timeseries but also other variables such as grey matter signal etc. This range spanns the nodes.
xyz = sio.loadmat('/data/grakas/Program/connprojects/conn_project_back2brain/results/preprocessing/ROI_Subject045_Condition001.mat')['xyz'][0,nstart:nend]
xyz2 = [np.squeeze(x) for x in xyz]
df2 = pd.DataFrame(xyz2, columns=['x','y','z'])
df2.to_csv(savedirs[0] + 'xyz.csv','\t') # save xyz coordinates of nodes

names = sio.loadmat('/data/grakas/Program/connprojects/conn_project_back2brain/results/preprocessing/ROI_Subject046_Condition001.mat')['names'][0,nstart:nend] # names of the nodes
names = np.array([x[0].split('.')[1] for x in names])

com = []
comnames = []
colors = []
for x in names:
    if 'cluster' in x: # subcortical
        com.append(7)
        comnames.append(x)
        colors.append('grey')
    if 'Vis' in x:
        com.append(0)
        comnames.append(x.split('_')[2] + ' ' + x.split('_')[-1])
        colors.append('Purple')
    if 'SomMot' in x:
        com.append(1)
        comnames.append(x.split('_')[2] + ' ' + x.split('_')[-1])
        colors.append('blue')
    if 'DorsAttn' in x:
        com.append(2)
        comnames.append(x.split('_')[2] + ' ' + x.split('_')[3])
        colors.append('green')
    if 'SalVentAttn' in x:
        com.append(3)
        comnames.append(x.split('_')[2] + ' ' + x.split('_')[3])
        colors.append('violet')
    if 'Limbic' in x:
        com.append(4)
        comnames.append(x.split('_')[2] + ' ' + x.split('_')[3])
        colors.append('palegoldenrod')
    if 'Cont' in x:
        com.append(5)
        comnames.append(x.split('_')[2] + ' ' + x.split('_')[3])
        colors.append('orange')
    if 'Default' in x:
        com.append(6)
        comnames.append(x.split('_')[2] + ' ' + x.split('_')[3])
        colors.append('crimson')

metadata = {}
metadata['yeo+tuan sub-network'] = com
metadata['yeo+tuan sub-network names'] = comnames
metadata['all subjects'] = []
metadata['cases'] = []
metadata['controls'] = []

correlation_measure = 'pearson' # Only used if use_conn_FC_matrices is False. Options: 'pearson', 'partial'
use_conn_FC_matrices = True
louvain = False
consensus_clustering = False
multiverse = False

th = 0.10 # threshold to threshold matrices

group_idx = [1 if x in cases_list else 0 for x in subs]


for subject_i, cond in enumerate(subs):

    if use_conn_FC_matrices: # Uses FC matrices for all subjects from CONN
        savedir = savedirs[2]

        # Get sorting vector

        # load nodes as sorted by CONN. Transforms from strings like 'schaefer.030_blabla' to 30_blabla.
        f = json.load(open('/data/grakas/Program/connprojects/conn_project_back2brain/results/secondlevel/SBC_02_ROI2ROI/AllSubjects/rest/exported_data.json','rb'))
        tmp = f['names']
        conn_sorted_extract_int = [node_name.split('.')[1] for node_name in tmp]
        del tmp

        # determing the vector to sort matrices
        sorting_vec = []
        for node_name in names:
            if node_name in conn_sorted_extract_int:
                idx = np.where(np.array(conn_sorted_extract_int)==node_name)[0]
                sorting_vec.append(idx[0])

        # get individual level matrices and sort
        unth_data = np.squeeze(nib.load('/data/grakas/Program/connprojects/conn_project_back2brain/results/secondlevel/SBC_02_ROI2ROI/AllSubjects/rest/exported_data.nii').get_fdata())
        corr = unth_data[:,:,subject_i][sorting_vec,:][:,sorting_vec]
        corr = np.nan_to_num(corr)

    else: # takes CONN denoised timeseries (and optionally teneto scrubbed (with interpolation)) timeseries.
        # load the preprocessed and scrubbed timeseries
        df = pd.read_csv(loaddir + subs[subject_i] + '/func/' + subs[subject_i] + '_run-1_task-rest_desc-conn-name-' + str(subject_i+1) + '_timeseries.tsv', '\t', index_col=0)
        stack.append(df)

        if correlation_measure=='pearson':
            # pearson corr
            corrdf = df.T.corr()
            corr = corrdf.values
            savedir = savedirs[0]
        elif correlation_measure=='partial':
            # partial corr (preregistration)
            corrdf = df.T.pcorr()
            corr = corrdf.T.pcorr().values
            savedir = savedirs[1]

    if subject_i==1:
        from netneurotools import plotting
        plotting.plot_mod_heatmap(corr, np.array(com), vmin=-1, vmax=1, cmap='coolwarm')

    ### Run steps for partial corr ##########################################
    # threshold net for participation coefficient
    corr = bct.threshold_proportional(corr, th) # also use [+- 0.5%]
    # corr[corr<0] = 0
    corrth = corr.copy()

    ### louvain. Prereg is ambiguous. Wrote both that we run louvain and that we use 7+1 networks (Schae/Tiun). Possibility: Multiverse: run pc/z analysis on both.
    if louvain:
        if consensus_clustering:
            # run n iterations and obtain consensus
            ci = [bct.community_louvain(corrth, gamma=gamma)[0] for n in range(100)]
            fig, ax = plt.subplots(1, 1, figsize=(6.4, 2))
            ax.imshow(ci, cmap='Set1')
            ax.set(ylabel='Assignments', xlabel='ROIs', xticklabels=[], yticklabels=[])

            from netneurotools import cluster
            com = cluster.find_consensus(np.column_stack(ci), seed=1234)
            plotting.plot_mod_heatmap(corrth, consensus, cmap='viridis')
        else:
            com = bct.community_louvain(corrth, gamma=gamma)[0]-1

    # within-module degree z
    z = bct.module_degree_zscore(corr, com) # unthresholded adjacency according to preregistration
    # participation coefficient
    pc = bct.participation_coef(corrth, com)
    #########################################################################

    ### computations for hypothesis III #####################################
    # cases/controls
    mxpc = [pc[np.array(com)==x].max() for x in range(8)]
    mxz = [z[np.array(com)==x].max() for x in range(8)]
    mdpc = [np.median(pc[np.array(com)==x]) for x in range(8)]
    mdz = [np.median(z[np.array(com)==x]) for x in range(8)]

    maxPC[subject_i] = mxpc
    maxZ[subject_i] = mxz

    medianPC[subject_i] = mdpc
    medianZ[subject_i] = mdz

    if subs[subject_i] in cases_list:
        s = np.where(np.array(cases_list)==subs[subject_i])[0][0]
        maxPC_cases[s] = mxpc
        maxZ_cases[s] = mxz
        medianPC_cases[s] = mdpc
        medianZ_cases[s] = mdz
    elif subs[subject_i] in controls_list:
        k = np.where(np.array(controls_list)==subs[subject_i])[0][0]
        maxPC_controls[k] = mxpc
        maxZ_controls[k] = mxz
        medianPC_controls[k] = mdpc
        medianZ_controls[k] = mdz

    # # save indices for each node for use later in visualization
    # maxPCidx[subs[ii]] = mxpc
    # maxZidx[subs[ii]] = mxz
    #
    # medianPCidx[subs[ii]] = mdpc
    # medianZidx[subs[ii]] = mdz

    #########################################################################

    # collect data
    # Networks
    net[subject_i,:,:] = corrth # note only thresholded (used for PC). Add unthresh (used for Z)
    netunthresholded[subject_i,:,:] = corr

    # PC/z
    participation_coef[subject_i,:] = pc
    module_degree_zscore[subject_i,:] = z

    # retreive metadata
    metadata['all subjects'].append(subs[subject_i])

    # Save thresholded network
    if subs[subject_i] in cases_list:
        s = np.where(np.array(cases_list)==subs[subject_i])[0][0]
        cases[s,:,:] = corrth
        cases_unthresholded[s,:,:] = corr
        # PC/z
        pc_cases[s,:] = pc
        z_cases[s,:] = z

        metadata['cases'].append(subs[subject_i])
    elif subs[subject_i] in controls_list:
        k = np.where(np.array(controls_list)==subs[subject_i])[0][0]
        controls[k,:,:] = corrth
        controls_unthresholded[k,:,:] = corr
        # PC/z
        pc_controls[k,:] = pc
        z_controls[k,:] = z

        metadata['controls'].append(subs[subject_i])

if multiverse:
    savedir = savedir + '/multiverse_results/' + 'threshold-' + str(th) + '/'
#    if ~os.path.isdir(savedir):
#        os.makedirs(savedir)

# save data corr
np.save(savedir + 'FC_group-all.npy',net)
np.save(savedir + 'FC_group-cases.npy',cases)
np.save(savedir + 'FC_group-controls.npy',controls)
np.save(savedir + 'modz_group-all.npy',module_degree_zscore)
np.save(savedir + 'modz_group-cases.npy',z_cases)
np.save(savedir + 'modz_group-controls.npy',z_controls)
np.save(savedir + 'PC_group-all.npy',participation_coef)
np.save(savedir + 'PC_group-cases.npy',pc_cases)
np.save(savedir + 'PC_group-controls.npy',pc_controls)

np.save(savedir + 'maxPC_group-cases.npy',maxPC_cases)
np.save(savedir + 'maxZ_group-cases.npy',maxZ_cases)
np.save(savedir + 'medianPC_group-cases.npy',medianPC_cases)
np.save(savedir + 'medianZ_group-cases.npy',medianZ_cases)

np.save(savedir + 'maxPC_group-controls.npy',maxPC_controls)
np.save(savedir + 'maxZ_group-controls.npy',maxZ_controls)
np.save(savedir + 'medianPC_group-controls.npy',medianPC_controls)
np.save(savedir + 'medianZ_group-controls.npy',medianZ_controls)

np.save(savedir + 'maxPC_group-all.npy',maxPC)
np.save(savedir + 'maxZ_group-all.npy',maxZ)
np.save(savedir + 'medianPC_group-all.npy',medianPC)
np.save(savedir + 'medianZ_group-all.npy',medianZ)


# save metadata
metadata['info'] = 'This is information is about the order of participants in the three numpy files. "all_subjects" refers to the net.npy files and points to the position of each subject (according to conn toolbox). The same for cases and controls which points to the order of participant in each group in the sub,node,node numpy files.'
metadata['Derived functional connectivity'] = 'The numpy files contain participant-level functional connectivity derived using ' + correlation_measure + ' correlation with import pingouin. In the pearsonr folder are found everything that was done with partial correlation but computed with pearson correlation: timeseries.corr(). Datasaved in /conn/ is the already computed bivariate correlations in conn toolbox.'
metadata['PC/z'] = 'Participation coefficient and within-module degree z-score computed using pybct and thresholded (for PC) and unthresholded (for z) according to preregistration. Results for pearson correlation are save in /data/grakas/kirurgi/analysis/cartographic_profile/pearsonr'
metadata['multiverse'] = 'For partial and pearson correlation respectively, there are certain choices that can be made, e.g. thresholding at different levels. According to preregistration, we present results at threshold-level 10% but present also results for +-5%.'
metadata['thresohld'] = th
with open(savedir + '/metadata.json','w') as f:
    json.dump(metadata, f)


## Visualizations (diagnostics)
import matplotlib


### with FMWH==6:
FMWH = False
if FMWH:
    netth_avg = sio.loadmat('/data/grakas/res/CONN_netth_conv.mat')['netth_conv']
    diff = sio.loadmat('/data/grakas/res/CONN_netth_diff_conv.mat')['netth_diff_conv']


netth_avg = np.mean(net,axis=0)
netth_cases_avg = np.mean(cases,axis=0)
netth_controls_avg = np.mean(controls,axis=0)
diff = netth_cases_avg - netth_controls_avg

# leave one row empty
z = np.zeros((10+netth_avg.shape[0],10+netth_avg.shape[0]))
z_group = np.zeros((10+netth_avg.shape[0],10+netth_avg.shape[0]))
z[:-10,10:] = netth_avg

# Used for plotting patches along the x axis and y-axis (transposed)
mask = np.zeros((10,432+10))
mask[mask==0]=999
z[432:,:]= mask
z[:,:10] = mask.T

z_group[:-10,10:] = diff
z_group[432:,:] = mask
z_group[:,:10]= mask.T

cmap = sns.diverging_palette(230, 20, as_cmap=True)

fig,axes = plt.subplots(1,2,figsize=(12,6))

sns.heatmap(z,mask=z==999,square=True,cbar_kws={'shrink':.3,'use_gridspec':False,'location':'top'},ax=axes[0])
sns.heatmap(z_group,cmap=cmap,mask=z_group==999,square=True,cbar_kws={'shrink':.3,'use_gridspec':False,'location':'top'},ax=axes[1])

axes[0].collections[0].colorbar.set_label('Mean FC $(z)$',fontsize='large')
axes[0].collections[0].colorbar.set_ticks([0.0,1.3])
#ax.collections[0].colorbar.set_ticklabels([r'Int',r'Seg'])
# ax.collections[0].colorbar.ax.tick_params(labelsize=14)
# ax.collections[0].colorbar.ax.get_xaxis().set_label_coords(0.5,2)
axes[1].collections[0].colorbar.set_label('Diff FC (Cases - Controls)',fontsize='large')
axes[1].collections[0].colorbar.set_ticks([-0.3,0.3])

axes[0].set_yticks([])
axes[0].set_yticks([])
axes[0].set_xticks([])
axes[1].set_yticks([])
axes[1].set_yticks([])
axes[1].set_xticks([])

# x axis
y=432
height=9
width=1
alpha = [0.4,0.5,0.6,0.7,0.8]
for ii in [0,1]:
    for step in range(432):
        axes[ii].add_patch(matplotlib.patches.Rectangle((10+step,y), width,height, color=colors[step],alpha=0.25))

# y axis
y=0
height=1
width=9
# only net
for ii in [0,1]:
    for step in range(432):
        axes[ii].add_patch(matplotlib.patches.Rectangle((y,step), width, height, color=colors[step],alpha=0.25))


### Check PC/z
# colors = ['purple','blue','green','violet','palegoldenrod','orange','crimson']


# participation coef
mean = np.mean(participation_coef,axis=0)
std = np.std(participation_coef,axis=0)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xticks(range(432))
for ii in range(432):
    ax.add_patch(matplotlib.patches.Rectangle((ii,-0.1),1,1,color=colors[ii],alpha=0.1))
ax.set_xticks([])
plt.plot(mean,color='black')
plt.fill_between(range(432),mean+std,mean-std,color='black',alpha=0.3)
sns.despine(bottom=True)
plt.ylabel('Mean participation coefficient')

# Z
mean = np.mean(module_degree_zscore,axis=0)
std = np.std(module_degree_zscore,axis=0)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xticks(range(432))
for ii in range(432):
    ax.add_patch(matplotlib.patches.Rectangle((ii,-3),1,5,color=colors[ii],alpha=0.1))
ax.set_xticks([])
plt.plot(mean,color='black')
plt.fill_between(range(432),mean+std,mean-std,color='black',alpha=0.3)
sns.despine(bottom=True)
plt.ylabel('Mean module degree z')

# plt.fill_between(range(219),DMN_mean+DMN_sd,DMN_mean-DMN_sd,color='crimson',alpha=0.25)

########## End ############

###############################################################################################
# 10.
### statistics section ########################################################################
# for median/max PC/z for each networks, run permutations. Shift group-membership.
# For each network, we permute participant labels. Null hypothesis is that mean of the measure
# does not differ between groups. Therefore, it does not matter if permute participants' group membership
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests

plt.ion()

def permwr(dat1, dat2, exchange='subjects',pnum=10000, tail=2, ):
    """
    permutation without replacement
    """
    if exchange == 'subjects':

        permdist = np.zeros(pnum+1) # to add original observation
        for i in range(0, pnum):
            permgroup = np.concatenate((dat1, dat2))

            permgroup1 = np.random.choice(permgroup,size=len(dat1))
            permgroup2 = np.random.choice(permgroup,size=len(dat2))

            if tail == 2:
                permdist[i] = abs(np.mean(permgroup1) - np.mean(permgroup2))
            elif tail == 1:
                permdist[i] = np.mean(permgroup1) - np.mean(permgroup2)

        if tail == 2:
            empdiff = abs(dat1.mean() - dat2.mean())

        elif tail == 1:
            empdiff = dat1.mean() - dat2.mean()

        permdist[pnum] = empdiff

        permdist = np.sort(permdist)
        p_value = (1+sum(permdist > empdiff)) / (pnum+1) # see  Phipson and Smyth, Permutation P-values should never be zero

        c = np.sqrt((p_value*(1-p_value)) / (pnum+1))
        confint = [np.round_(p_value-c,decimals=4), np.round_(p_value+c,decimals=4)]

        return p_value, permdist

def set_sig_star(ax,pvals,corrected_pvals=None):
    for ii, p in enumerate(pvals):
        if p<0.05:
            star = '*'
            if corrected_pvals is not None:
                if corrected_pvals[0]<0.05:
                    star = '***'

            xticklabels = ax.get_xticklabels()
            to_change = xticklabels[ii].get_text()
            changed = to_change + '\n' + star
            new_xticklabels = xticklabels.copy()
            new_xticklabels[ii] = changed

            ax.set_xticklabels(new_xticklabels)


if ~os.path.exists(savedir + '/figures/'):
    os.mkdir(savedir + '/figures/')
else:
    print('Folder exists')


medianPC_cases = np.load('/data/grakas/kirurgi/analysis/cartographic_profile/conn/medianPC_group-cases.npy')
medianZ_cases = np.load('/data/grakas/kirurgi/analysis/cartographic_profile/conn/medianZ_group-cases.npy')
maxPC_cases = np.load('/data/grakas/kirurgi/analysis/cartographic_profile/conn/maxPC_group-cases.npy')
maxZ_cases = np.load('/data/grakas/kirurgi/analysis/cartographic_profile/conn/maxZ_group-cases.npy')

medianPC_controls = np.load('/data/grakas/kirurgi/analysis/cartographic_profile/conn/medianPC_group-controls.npy')
medianZ_controls = np.load('/data/grakas/kirurgi/analysis/cartographic_profile/conn/medianZ_group-controls.npy')
maxPC_controls = np.load('/data/grakas/kirurgi/analysis/cartographic_profile/conn/maxPC_group-controls.npy')
maxZ_controls = np.load('/data/grakas/kirurgi/analysis/cartographic_profile/conn/maxZ_group-controls.npy')

cases_results = [medianPC_cases, medianZ_cases, maxPC_cases, maxZ_cases]
controls_results = [medianPC_controls, medianZ_controls, maxPC_controls, maxZ_controls]

savedir = '/data/grakas/kirurgi/analysis/cartographic_profile/conn/'

# get covariates
df_cov = pd.read_csv('/data/grakas/kirurgi/analysis/cartographic_profile_results_only_covariates.csv',index_col=False)
df_cov.drop('Unnamed: 0',1,inplace=True)
covariates = ['Group','VAS_Back_Correct', 'ODI_Score_Exam','MCS_Exam', 'L4_L5_Pf',
       'L5_S1_Pf','mean_trunk_angle','mean_headangle']

df_cov = df_cov[covariates]
df_cov.replace({999.0:np.nan,9999.0:np.nan},inplace=True)

cases_results_numpy = np.concatenate(cases_results,axis=1)
controls_results = np.concatenate(cases_results,axis=1)

all_results = np.concatenate([cases_results_numpy,controls_results],axis=0)

df_brain = pd.DataFrame(all_results)
df_brain.columns
columns = ['MedianPC_sub', 'MedianPC_vis', 'MedianPC_sm',
       'MedianPC_da', 'MedianPC_sa', 'MedianPC_lim', 'MedianPC_fp',
       'MedianPC_dmn', 'MaxPC_sub', 'MaxPC_vis', 'MaxPC_sm', 'MaxPC_da',
       'MaxPC_sa', 'MaxPC_lim', 'MaxPC_fp', 'MaxPC_dmn', 'MedianZ_sub',
       'MedianZ_vis', 'MedianZ_sm', 'MedianZ_da', 'MedianZ_sa', 'MedianZ_lim',
       'MedianZ_fp', 'MedianZ_dmn', 'MaxZ_sub', 'MaxZ_vis', 'MaxZ_sm',
       'MaxZ_da', 'MaxZ_sa', 'MaxZ_lim', 'MaxZ_fp', 'MaxZ_dmn']

df_brain.columns = columns

df_all = pd.concat([df_cov[covariates],df_brain],axis=1)

def test_measures(df):
    """
    X contains dependent variables (median/max PC/z)
    Y contains covaraites
    """
    from netneurotools.stats import residualize
    n_measures = 4
    n_nets = 8
    N = n_measures, n_nets

    # remove rows with nan
    df.dropna(inplace=True)

    # residualize
    metadata = {}
    uncorr_pvals = []
    for col_i, col_name in enumerate(columns):
        # temporary df_cov. Attach brain network for a measure (i.e. col_name) to it.
        print(col_name)
        tmp_df = df[covariates].copy()
        tmp_df.drop('Group',1,inplace=True)

        # dependent variable
        Y = df[col_name].values

        # residualize
        res_Y = residualize(tmp_df.values, Y)

        # separate groups
        res_Y_cases = Y[df['Group']==1]
        res_Y_controls = Y[df['Group']==2]

        # statistical test
        pval, permdist = permwr(res_Y_cases, res_Y_controls)

        uncorr_pvals.append(pval)

    corr_pvals = multipletests(uncorr_pvals,method='fdr_bh')[1]

    metadata['corrected_pvals'] = {'corrected_pvals':list(corr_pvals)}

    with open(savedir + '/stats.json','w') as f:
        json.dump(metadata, f)

    return uncorr_pvals, metadata

# Originally, looped through each measure and each network and made group comparisons.
def test_measures():
    metadata = {}
    uncorr_pvals = []

    for ii, measure in enumerate(['maxPC','maxZ','medianPC','medianZ',]):
        dfs = []
        dfY = []
        metadata[measure] = []
        pvals = []
        for N, network in enumerate(['Vis','SM','DA','SA','Limbic','FP','DMN','Subcortical']):
            test_cases = cases_results[ii]
            test_controls = controls_results[ii]

            if np.where(test_cases==0)[0].size>0:
                print('cases')
            if np.where(test_controls==0)[0].size>0:
                print('controls')

            pval, permdist = permwr(test_controls[:,N], test_cases[:,N])

            # save info
            metadata[measure].append( {'network':network,'p-value': pval })
            pvals.append(pval)
            uncorr_pvals.append(pval)

            # save figures
            df = pd.DataFrame()
            df[measure] = test_cases[:,N]
            df['group'] = np.repeat('Cases',23)
            df['network'] = np.repeat(network, 23)
            df2 = pd.DataFrame()
            df2[measure] = test_controls[:,N]
            df2['group'] = np.repeat('Controls',23)
            df2['network'] = np.repeat(network, 23)
            dfX = pd.concat([df,df2],axis=0)
            dfs.append(dfX)

        dfY = pd.concat(dfs,axis=0)

        plt.figure(figsize=(12, 6))
        ax = sns.boxplot(y=dfY[measure], x=dfY['network'], hue=dfY['group'],boxprops=dict(alpha=0.8))
        leg = ax.legend(loc = 'lower right')
        for x in leg.legendHandles:
            x.set_alpha(0.8)

        # set_sig_star(ax,pvals)
        # set_sig_star(ax,pvals)
        # set_sig_star_fdr(ax,corr_pvals)

        if ii>1:
            leg.remove()

        plt.xlabel('')
        sns.despine()
        plt.savefig(savedir + '/figures/' + measure + '.png')
        plt.close()

    corr_pvals = multipletests(uncorr_pvals,method='fdr_bh')[1]
    metadata['corrected_pvals'] = {'corrected_pvals':list(corr_pvals)}

    with open(savedir + '/stats.json','w') as f:
        json.dump(metadata, f)

    return pvals, corr_pvals, metadata



### Run multiple t-tests
z_cases = np.load('/data/grakas/kirurgi/analysis/cartographic_profile/conn/modz_group-cases.npy')
z_controls = np.load('/data/grakas/kirurgi/analysis/cartographic_profile/conn/modz_group-controls.npy')
pc_cases = np.load('/data/grakas/kirurgi/analysis/cartographic_profile/conn/PC_group-cases.npy')
pc_controls = np.load('/data/grakas/kirurgi/analysis/cartographic_profile/conn/PC_group-controls.npy')

T_vals = []
p_vals = []
test_obj = [z_cases, z_controls]
for node in range(pc_cases.shape[-1]):
    A = test_obj[0][:,node]
    B = test_obj[1][:,node]
    T,p = scipy.stats.ttest_ind(A, B)
    T_vals.append(T)
    p_vals.append(p)

p_vals_corr = multipletests(p_vals,method='fdr_bh')[1]

# Plot p-values
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xticks(range(432))
for ii in range(432):
    ax.add_patch(matplotlib.patches.Rectangle((ii-0.5,-0.1),1,1.4,color=colors[ii],alpha=0.1))

ax.set_xticks([])
plt.plot(-np.log10(p_vals),color='black')
sns.despine(bottom=True)
plt.ylabel(r'$-log_{10}(p)$')
plt.xlabel('Nodes')

plt.hlines(1.3,0,432,'black','--')

plt.plot(-np.log10(p_vals_corr),color='crimson',alpha=0.65)

plt.legend(['Uncorrected',r'$FDR_{BH}$'],frameon=False)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xticks(range(432))
for ii in range(432):
    ax.add_patch(matplotlib.patches.Rectangle((ii-0.5,-1),1,2,color=colors[ii],alpha=0.1))

ax.set_xticks([])
plt.plot(T_vals,color='black')
sns.despine(bottom=True)
plt.ylabel(r'$T$')
plt.xlabel('Nodes')

plt.legend(['Cases > Controls'],frameon=False,fontsize=14)
plt.hlines(0,0,432,'black','--')

### Ancova
pc_all = np.load('/data/grakas/kirurgi/analysis/cartographic_profile/conn/PC_group-all.npy')
z_all = np.load('/data/grakas/kirurgi/analysis/cartographic_profile/conn/modz_group-all.npy')
group  = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,
       2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2,
       2, 2])

T_vals = []
p_vals = []
test_obj = z_all.copy()
for node_i in range(pc_cases.shape[-1]):
    A = test_obj[:,node_i]
    B = test_obj[:,node_i]
    T,p = scipy.stats.ttest_ind(A, B)
    T_vals.append(T)
    p_vals.append(p)

p_vals_corr = multipletests(p_vals,method='fdr_bh')[1]



############################
### Exploratory analysis ###
############################
import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
import random
import bct
import os

##############################
### network-based statistics #
##############################
positive = np.greater
negative = np.less

alpha = 0.05
nsubs = 46

x = cases_data.copy()
x = x.transpose(1,2,0)
x[positive(x,0)] = 0
y = controls_data.copy()
y = y.transpose(1,2,0)
y[positive(y,0)] = 0

t = stats.distributions.t.ppf(1 - alpha, nsubs - 1)
p,adj,null = bct.nbs_bct(x,y,t)

######### end NBS ############

