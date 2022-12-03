# Validity check of the two images
def validity(left, right):
    for i in range(0, left.shape[0], 1):
        for j in range(0, left.shape[1], 1):
            if left[i,j] != right[i,j]:
                left[i,j] = 0
            if right[i, j] != left[i, j]:
                right[i, j] = 0

    return left, right