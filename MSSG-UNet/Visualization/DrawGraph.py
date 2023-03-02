import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.cm as cmx
import scipy.io as sio
from scipy.special import softmax

dataset_name=['indian_','paviaU_','salinas_'][2]
coder_type=['encoders','decoders'][0]


A_edcoders=np.load(dataset_name+'A_encoders.npz')
A_decoders=np.load(dataset_name+'A_decoders.npz')


if dataset_name == 'indian_':
    segs = sio.loadmat('segments\\segmentmaps' + 'indian.mat')
    segs = segs['segmentmaps']
if dataset_name == 'paviaU_':
    segs = sio.loadmat('segments\\segmentmaps' + 'paviau.mat')
    segs = segs['segmentmaps']
if dataset_name == 'salinas_':
    segs = sio.loadmat('segments\\segmentmaps' + 'salinas.mat')
    segs = segs['segmentmaps']



print('computing..')
scale=15.0
dpi=400

# plt.scatter(coo[:,0], coo[:,1], marker = '.',color = 'red' ,s=100)

for idx in range(4):
    if coder_type == 'encoders': A_1 = A_edcoders["a"+str(idx+1)]
    if coder_type == 'decoders': A_1 = A_decoders["a"+str(4-idx)]
    
    # Test
    # A_1=softmax(A_1,axis=-1)
    
    seg=segs[idx]
    coordinates = []
    for i in range(A_1.shape[0]):
        temp = np.where(seg == int(i))
        coordinates.append([np.average(temp[0]), np.average(temp[1])])
    
    coo = np.array(coordinates)
    
    A=A_1
    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    fig.set_size_inches(seg.shape[0] * scale / dpi, seg.shape[1] * scale / dpi)
    jet = cm = plt.get_cmap('jet')
    # jet = cm = plt.get_cmap('OrRd')
    # cNorm  = colors.Normalize(vmin=np.min(np.reshape(A,[-1])[np.where(np.reshape(A,[-1])!=0)[0]]), vmax=np.max(A))
    cNorm = colors.Normalize(vmin=0, vmax=1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    print(scalarMap.get_clim())
    
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i,j]<0.000001: continue
            if i==j: continue
            tt1=coo[i]
            tt2=coo[j]
            colorVal = scalarMap.to_rgba(A[i,j])
            plt.plot([tt1[0],tt2[0]],[tt1[1],tt2[1]],linewidth=2,c=colorVal)#,norm=norm)
    plt.margins(0,0)
    plt.savefig("adjacency\\"+dataset_name+"_"+coder_type+"_a"+str(idx+1),transparent=True, pad_inches = 0)
    plt.show()
    
# if 'A_2':
#     A=A_2
#     fig, ax = plt.subplots()
#     ax.set_axis_off()
#     ax.xaxis.set_visible(False)
#     ax.yaxis.set_visible(False)
#     plt.gca().xaxis.set_major_locator(plt.NullLocator())
#     plt.gca().yaxis.set_major_locator(plt.NullLocator())
#     plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
#     fig.set_size_inches(seg.shape[0] * scale / dpi, seg.shape[1] * scale / dpi)
#     jet = cm = plt.get_cmap('jet')
#     # jet = cm = plt.get_cmap('OrRd')
#     # cNorm  = colors.Normalize(vmin=np.min(np.reshape(A,[-1])[np.where(np.reshape(A,[-1])!=0)[0]]), vmax=np.max(A))
#     cNorm = colors.Normalize(vmin=-1, vmax=1)
#     scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
#     print(scalarMap.get_clim())
#
#     for i in range(A.shape[0]):
#         for j in range(A.shape[1]):
#             if A[i,j]==0: continue
#             if i==j: continue
#             tt1=coo[i]
#             tt2=coo[j]
#             colorVal = scalarMap.to_rgba(A[i,j])
#             plt.plot([tt1[0],tt2[0]],[tt1[1],tt2[1]],linewidth=2,c=colorVal)#,norm=norm)
#     plt.margins(0,0)
#     plt.savefig(dataset_name+'A_2',transparent=True, pad_inches = 0)
#     plt.show()