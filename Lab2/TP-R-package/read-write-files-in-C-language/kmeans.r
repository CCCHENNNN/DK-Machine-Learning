

D <- read.table("/Users/hchen/Desktop/DK/Machine\ Learning/Lab2/TP-R-package/read-write-files-in-C-language/input.txt")
print(D[3,])
print(ncol(D))
kmeans <- function(D, k=7, threshold=1) {

	# Random centers
	center.ids = sample(1:nrow(D), size=k)#无放回抽样 抽出两个行号
	centers = D[center.ids,]
	print(centers)
	divergence = 2 * threshold

	while (divergence > threshold) {
		# Euclidean
		E = NULL
		for (i in 1:k) {
			# print(row)
			print(centers[i])
			E = cbind(E, apply(D, 1, function(row) { abs(row - centers[i]) }))
			# print(E)
			#第一个参数是指要参与计算的矩阵；
			#第二个参数是指按行计算还是按列计算，1——表示按行计算，2——按列计算；
			#第三个参数是指具体的运算参数。
		}
		

		id = apply(E, 1, function(row) { which.min(row) })
		print(id[1])
		old = centers
		# print(old)
		for (i in 1:k) {
			indices = which(id == i)
			# print(indices)
			centers[i] = mean(D[indices,])
		}
		print(centers)

		# Frobenius norm
		print(old)
		# print(centers)
		print(sum((old - centers)^2))
		divergence = sum(abs(old - centers))
		cat("Div: ", divergence, "\n")
	}

	ret = list()
	ret$centers = centers
	ret$id = id
	ret$divergence = divergence
	idd = id *50
	write(idd, file = "/Users/hchen/Desktop/DK/Machine\ Learning/Lab2/TP-R-package/read-write-files-in-C-language/output2.txt",sep = "\n")
	# print(ret)
	return (ret)
}

kmeans(D, k=5, threshold=1)
# Internal group distances
# External group distances
#
#
#
#












