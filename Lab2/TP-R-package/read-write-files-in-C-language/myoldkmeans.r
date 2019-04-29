Data <- read.table("/Users/hchen/Desktop/DK/Machine\ Learning/Lab2/TP-R-package/read-write-files-in-C-language/input.txt")
print(Data[2,])
getCenters <- function(Data,k){
	centers <- matrix(0,nrow=k,ncol=1)
  centersRandom <- sample(min(Data):max(Data), size=k)
  centers = cbind(centersRandom)
  print(centers)
	centers
}

getIndexMatrix <- function(Data,k,centers){
  indexMatrix <- matrix(0, nrow = nrow(Data), ncol = 2)
  for(i in 1:nrow(Data)){
    initialDistance <- 10000
    for(j in 1:k){
      currentDistance <- abs(Data[i,] - centers[j,])
      if(currentDistance < initialDistance){
        initialDistance <- currentDistance
        indexMatrix[i,1] <- j
        indexMatrix[i,2] <- currentDistance
      }
    }
  }
  indexMatrix
}

changeCenters <- function(Data,indexMatrix,k,centers){
  for(i in 1:k){
    clusterMatrix <- Data[indexMatrix[,1] == i,]
    clusterMatrix <- as.matrix(clusterMatrix)
    if(nrow(clusterMatrix) > 0){
      centers[i,] <- colMeans(clusterMatrix)
    }
    else{
      centers[i,] <- centers[i]
    }
  }
  centers
}

mykmeans <- function(Data, k=5, threshold=1){


	centers = getCenters(Data,k)
	print(centers)
	divergence = 2 * threshold

	while(divergence > threshold){
    indexMatrix = getIndexMatrix(Data,k,centers)
    old = centers
    centers = changeCenters(Data,indexMatrix,k,centers)
    divergence = sum(abs(old - centers))
    print(divergence)

	
	# indexMatrix <- matrix(0,nrow=nrow(Data),ncol=2)
	# for(i in 1:nrow(Data)){ 
 #      initialDistance <- 10000 
 #      # previousCluster <- indexMatrix[i,1]

 #      #遍历所有的类，将该数据划分到距离最近的类
 #      for(j in 1:k){ 
 #        currentDistance <- abs(Data[i,]-centers[j,])
 #        if(currentDistance < initialDistance){
 #           initialDistance <- currentDistance 
 #           indexMatrix[i,1] <- j 
 #           indexMatrix[i,2] <- currentDistance 
 #        } 
 #      }
 #    }
  #   old = centers
  #   for(m in 1:k){
  #   clusterMatrix <- Data[indexMatrix[,1]==m,] 
  #   clusterMatrix <- as.matrix(clusterMatrix)
  #   if(nrow(clusterMatrix)>0){ 
  #     centers[m,] <- colMeans(clusterMatrix) 
  #   } 
  #   else{
  #     centers[m,] <- centers[m,] 
  #   }    
  # }
  # print(centers)
  # divergence = sum(abs(old - centers))
  # print(divergence)
}

idd = as.integer(centers[indexMatrix[,1],])
aaa = centers[indexMatrix[,1],]
write(aaa, file = "/Users/hchen/Desktop/DK/Machine\ Learning/Lab2/TP-R-package/read-write-files-in-C-language/output4.txt",sep = "\n")
	write(idd, file = "/Users/hchen/Desktop/DK/Machine\ Learning/Lab2/TP-R-package/read-write-files-in-C-language/output3.txt",sep = "\n")
	# print(ret)

}
mykmeans(Data,k=5,threshold=1)
# getCenters(Data,k=5)