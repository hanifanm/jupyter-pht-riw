from matplotlib import pyplot as plt
import numpy as np
import cv2

N = 100
P = 70

img = cv2.imread('./maudy.png', 0)
img = cv2.resize(img, (N,N))
imvec = np.reshape(np.array(img), (N*N,1))

def imshow(imgdata):
    plt.imshow(imgdata, cmap = 'gray')
    plt.show()

# CREATE MATRIX W

print('create matrix w ...')

matN = np.ones((2 * P + 1, 2 * P + 1), dtype='float32')
matM = np.ones((2 * P + 1, 2 * P + 1), dtype='float32')
for i in range(-P, P+1):
    matN[i,:] = i
    matM[:,i] = i
matN = matN.reshape(((2 * P + 1) * (2 * P + 1), 1))
matN = np.repeat(matN, N * N, axis=1)
matM = matM.reshape(((2 * P + 1) * (2 * P + 1), 1))
matM = np.repeat(matM, N * N, axis=1)

matI = np.ones((N,N), dtype='float32')
matK = np.ones((N,N), dtype='float32')
for i in range(N):
    matI[i,:] = i
    matK[:,i] = i
matI = matI.reshape((1, N * N))
matI = np.repeat(matI, (2 * P + 1) * (2 * P + 1), axis=0)
matK = matK.reshape((1, N * N))
matK = np.repeat(matK, (2 * P + 1) * (2 * P + 1), axis=0)

matY = (2 * matI + 1 - N) / N
matX = (2 * matK + 1 - N) / N
matR = np.sqrt(matX * matX + matY * matY)
matT = np.arctan2(matY, matX)
matW = 4 / (np.pi * N * N) * np.exp(-2 * np.pi * matN * matR * matR * 1j - matM * matT * 1j)

matC = matR <= 1
matW = matC * matW

# CREATE MATRIX V

print('create matrix v ...')

matN = np.ones((2 * P + 1, 2 * P + 1), dtype='float32')
matM = np.ones((2 * P + 1, 2 * P + 1), dtype='float32')
for i in range(-P, P+1):
    matN[i,:] = i
    matM[:,i] = i
matN = matN.reshape((1, (2 * P + 1) * (2 * P + 1)))
matN = np.repeat(matN, N * N, axis=0)
matM = matM.reshape((1, (2 * P + 1) * (2 * P + 1)))
matM = np.repeat(matM, N * N, axis=0)

matI = np.ones((N,N), dtype='float32')
matK = np.ones((N,N), dtype='float32')
for i in range(N):
    matI[i,:] = i
    matK[:,i] = i
matI = matI.reshape((N*N,1))
matI = np.repeat(matI, (2 * P + 1)*(2 * P + 1), axis=1)
matK = matK.reshape((N*N,1))
matK = np.repeat(matK, (2 * P + 1)*(2 * P + 1), axis=1)

matY = (2 * matI + 1 - N) / N
matX = (2 * matK + 1 - N) / N
matR = np.sqrt(matX * matX + matY * matY)
matT = np.arctan2(matY, matX)
matV = np.exp(2 * np.pi * matN * matR * matR * 1j + matM * matT * 1j)

# CHECK ORTHOGONALITY

print('check orthogonality ...')

matVW = np.matmul(matV, matW)
matVW = np.abs(matVW)
matVW = np.floor(matVW)
print(matVW)