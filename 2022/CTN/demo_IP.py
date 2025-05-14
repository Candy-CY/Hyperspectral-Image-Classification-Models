def  readData(PCAID=0,WHITE=0):
     import numpy    as np
     import scipy.io as scio
     
     clas    = 16
     NUM     = 10249  #数据中的有效样本个数,12类10062,16类是10249
     datafile         = scio.loadmat("Indian_pines_corrected.mat")
     labelfile        = scio.loadmat("Indian_pines_gt.mat")
     data             = datafile['indian_pines_corrected']  #data.shape = [145,145,200]
     label            = labelfile['indian_pines_gt']        #label [145,145]

     data             = np.float32(data)
     label            = np.int32(label)
     data             = data/data.max()
#==============================================================================
     from sklearn.preprocessing import StandardScaler
     shapeor      = data.shape
     data         = data.reshape(-1, data.shape[-1])
     if PCAID == 1:
          from sklearn.decomposition import PCA
          num_components   = 64
          data         = PCA(n_components=num_components).fit_transform(data)
          shapeor      = np.array(shapeor)
          shapeor[-1]  = num_components
     if WHITE ==1:
          data         = StandardScaler().fit_transform(data)
     data         = data.reshape(shapeor)
# =============================================================================
    # 记录有效标记样本的坐标和标签，格式为（横坐标，纵坐标，标签值）
     rows,cols,bands  = data.shape
     samples          = np.zeros((NUM,3),dtype="int32")#NUM为有效样本点个数
     index            = 0
     for i in range(0,rows):
             for j in range(0,cols):
                 if label[i][j] != 0:
                      samples[index][0]     = i
                      samples[index][1]     = j
                      samples[index][2]     = label[i][j]-1#将有效标签写成0-15，共16类
                      index                 = index+1
     return data,samples,clas
 
def splitData(samples,c,r=0.1,rs=1024):
     from sklearn.model_selection    import train_test_split as ts
     import numpy
          
     #划分第0类的样本
     tmpS                      = samples[samples[:,-1]==0]
     numS                      = tmpS.shape[0]
     if r<1:
         num                       = int((numS)*r+0.5)#因为总样本中5，四舍五入
     else:
         num = r
         if num>numS:
             num = (numS+1)//2#因为总样本中5，四舍五入
     
     trainX,testX,trainY,testY = ts(tmpS,tmpS,train_size = num,random_state = rs)
     trainD                    = trainX
     testD                     = testX
     print(tmpS.shape[0],trainX.shape[0])
     
     #划分其他类别样本
     for i in range(1,c):
        tmpS                      = samples[samples[:,-1]==i]
        numS                      = tmpS.shape[0]
        if r<1:
            num                       = int((numS)*r+0.5)#因为总样本中5，四舍五入
        else:
            num = r
            if num>numS:
                num = (numS+1)//2#因为总样本中5，四舍五入
        trainX,testX,trainY,testY = ts(tmpS,tmpS,train_size = num,random_state = rs)
        trainD = numpy.vstack((trainD,trainX))    
        testD  = numpy.vstack((testD,testX))
        print(tmpS.shape[0],trainX.shape[0])
     
     return trainD,testD

def getData(data,trD,block=9):
     import numpy

     num             = trD.shape[0]
     rows,cols,bands = data.shape
     bk              = int(block//2)
     
     
     # 扩展填充
     tmpData                              = numpy.zeros((rows+bk*2,cols+bk*2,bands),dtype="float32")
     tmpData[bk:bk+rows,bk:bk+cols,:]     = data[:,:,:]
     tmpData[0:bk,:,:]                    = numpy.expand_dims(tmpData[bk,:,:],axis=0).repeat(bk,axis=0)
     tmpData[rows+bk:rows+2*bk,:,:]       = numpy.expand_dims(tmpData[bk+rows,:,:],axis=0).repeat(bk,axis=0)
     tmpData[:,0:bk,:]                    = numpy.expand_dims(tmpData[:,bk,:],axis=1).repeat(bk,axis=1)
     tmpData[:,cols+bk:cols+2*bk,:]       = numpy.expand_dims(tmpData[:,bk+cols,:],axis=1).repeat(bk,axis=1)    
     
     # 提取样本点
     trX             = numpy.zeros((num,block,block,bands),dtype="float32")
     trY             = numpy.zeros(num,dtype="int32")
          
     for i in range(num):
          posX             = trD[i,0]+bk
          posY             = trD[i,1]+bk
          trX[i,:,:,:]     = tmpData[(posX-bk):(posX+bk+1),(posY-bk):posY+bk+1,:]
          trY[i]           = trD[i,2]     
     return trX,trY
    
def Position(L):  
     import numpy as np2
     pos  = np2.zeros((L,L))
     xc   = L//2
     yc   = L//2
     for h in range(L):
          for k in range(L):
               pos[h,k]=np2.max([np2.abs(h-xc),np2.abs(k-yc)])+1
     print(pos)
     pos  = pos.reshape((1,L*L)) 
     return pos

#==============================================================================
from Transformer                import GELU,FeedForward
from keras_layer_normalization import LayerNormalization
from keras.models               import load_model
import numpy  as np
import keras
DB    =  'IP'
RATE  =  0.1
BLOCK =  15
    
    
data,samples,num_clas = readData(PCAID=1,WHITE=1)

# 1.get train data
trainD,testD          = splitData(samples,num_clas,r=RATE,rs=1)
trX,trY               = getData(data,trainD,block=BLOCK)
num,rows,cols,bands   = trX.shape
pos                   = Position(L=BLOCK)
X_pos                 = np.dot(np.ones((num,1)),pos)
X_pos                 = X_pos.reshape((num,BLOCK,BLOCK,1))


# 2.build model 
input_shape           = (rows,cols,bands)

model                 = load_model('IP_model.h5',custom_objects={'GELU':GELU,'FeedForward':FeedForward,'LayerNormalization':LayerNormalization})

teX,teY      = getData(data,testD,block=BLOCK)
X_pos2       = np.dot(np.ones((teX.shape[0],1)),pos)
X_pos2       = X_pos2.reshape((teX.shape[0],BLOCK,BLOCK,1))
    
X_test       = teX.reshape(-1,rows,cols,bands)
Y_test       = keras.utils.to_categorical(teY,num_clas)
score        = model.evaluate([X_test,X_pos2], Y_test)        
print('Test loss:', score[0])
print('Test accuracy:', score[1]) 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    