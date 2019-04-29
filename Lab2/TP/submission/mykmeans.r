########################################
# Employ K-Means to codify audio signals
########################################

# Read the textual integers file to "Data", "Data" is matrix with the col = 1
Data <- read.table("/Users/hchen/Desktop/DK/Machine\ Learning/Lab2/TP/xuedi_input.txt")

# To get the original k center points 
# k random integers from min(Data) to max(Data)
getCenters <- function(Data,k){
	centers <- matrix(0,nrow=k,ncol=1) # initialize centers matrix
  centersRandom <- sample(min(Data):max(Data), size=k)
  centers = cbind(centersRandom)
	centers
}

# To get a matrix which records the cluster each point of "Data" belongs and its distance to the cluster's center
# col1 = the number of cluster which corresponds to the index of the centers
# col2 = the closest distance
getIndexMatrix <- function(Data,k,centers){
  indexMatrix <- matrix(0, nrow = nrow(Data), ncol = 2) # initialize the matrix
  for(i in 1:nrow(Data)){
    initialDistance <- 10000 # initialize the distance with a large value
    for(j in 1:k){ 
      # for each point of "Data", traversing the centers and find one which is closest to it
      currentDistance <- abs(Data[i,] - centers[j,]) # calculate the distance
      if(currentDistance < initialDistance){ # to find the closest point
        initialDistance <- currentDistance
        indexMatrix[i,1] <- j
        indexMatrix[i,2] <- currentDistance
      }
    }
  }
  indexMatrix
}

# Update the centers for the next loop
changeCenters <- function(Data,indexMatrix,k,centers){
  for(i in 1:k){
    # "clusterMatrix" is a matrix which contains all the points of "Data" for different clusters
    clusterMatrix <- Data[indexMatrix[,1] == i,]
    clusterMatrix <- as.matrix(clusterMatrix)
    if(nrow(clusterMatrix) > 0){
      centers[i,] <- colMeans(clusterMatrix) # update the center with the average of the points' values
    }
    else{
      centers[i,] <- centers[i,] # if no point for the center, it doesn't change
    }
  }
  centers
}

# Main function of K-Means
mykmeans <- function(Data, k=5, threshold=1){
	centers = getCenters(Data,k)
	divergence = 2 * threshold # initialize the divergence

	while(divergence > threshold){ # the while loop work when divergence > threshold
    indexMatrix = getIndexMatrix(Data,k,centers) # get the record matrix
    old = centers
    centers = changeCenters(Data,indexMatrix,k,centers) # change the centers
    divergence = sum(abs(old - centers)) # update the value of divergence
    print(divergence)
  }

  # for each one of the k clusters, we change all the points' values into its center value
  # to make the output have only k values
  result = as.integer(centers[indexMatrix[,1],]) # change float to int
  # write the result into output.txt
  write(result, file = "/Users/hchen/Desktop/DK/Machine\ Learning/Lab2/TP/xuedi_output25.txt",sep = "\n")
}

mykmeans(Data,k=25,threshold=1)




