import torch

scaler = torch.tensor(7)
# to get demeansion of tensor 
print(scaler.ndim)

print("scaler:",scaler)
# Get the Python number within a tensor (only works with one-element tensors)
print("scaler dimension:",scaler.item())

#vector 
vector = torch.tensor([2,3])
print("vector:", vector)

# Check the number of dimensions of vector
print("vector dimension:",vector.ndim)
#can tell no of dimensin by no of open square brackets

#shape of the vector
print("vector shape",vector.shape)
#The above returns torch.Size([2]) which means our vector has a shape of [2]. This is because of the two elements we placed inside the square brackets ([2,3]).

#metrix
matrix = torch.tensor([
    [1,2],
    [3,4],
    ])

print("metrix;", matrix)

#metrix dimension
print("metrix dim:", matrix.dim)

#metrix shape
print("metrix shape", matrix.shape)


#tensor
tensor = torch.tensor([[
    [1,2,3,4],
    [5,6,7,8],
    [9,10,11,12],
    [13,14,15,16]
    ]])

print("tensor", tensor)

#tensor have 2 square bracket before the main ones and vecore have 1 and sclare dont 

#tensor shape
print("tensor shape:",tensor.shape)

#tensor dimension
print("tensor dimension", tensor.dim)

#scalars and vectors denoted as lowercase letters such as y or a. And matrices and tensors denoted as uppercase letters such as X or W.
#tensor is  	an n-dimensional array of numbers
# scalars are 0 dimensional and vecotor are 1 dimensonal and tensor are n dimensonal but genrally 2

#random tensor with size of (2,4)
random_tensor = torch.rand(size=(2,4))

print("random tensor", random_tensor)

# Create a random tensor of size (224, 224, 3)
random_image_size_tensor = torch.rand(size=(224, 224, 3))
print("random_img_siz shape:", random_image_size_tensor.shape,"rad_img_siz_dim:", random_image_size_tensor.ndim)


