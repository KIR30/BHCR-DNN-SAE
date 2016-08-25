library(“deepnet”, lib.loc=”~/R/x86_64-pc-linux-gnu-library/3.0″)
#Data Loading 
train <- read.delim("~/bangla_data/train.csv", header=FALSE)
test <- read.delim("~/bangla_data/test.csv", header=FALSE)
trainlabel <- read.table("~/bangla_data/trainlabel.csv", quote="\"", comment.char="")
testlabel <- read.table("~/bangla_data/testlabel.csv", quote="\"", comment.char="")

#Data Preprocessing 
train = as.matrix(train)
test = as.matrix(test)

trainlabel = as.vector(trainlabel)
testlabel = as.as.vector(testlabel)

#Train deep neural network with weights initialized by stack autoencoder 

dnn =sae.dnn.train(train, trainlabel, hidden = c(10,10), activationfun = "sigm", learningrate = 0.8, 
              momentum = 0.5, learningrate_scale = 1, output = "sigm", sae_output = "linear", 
              numepochs = 50, batchsize = 100, hidden_dropout = 0, visible_dropout = 0)

#test
nn.test(dnn, test, testlabel)


