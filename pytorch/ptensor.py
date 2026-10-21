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

# Create a random tensor of size (224, 224, 3) random tensor in the common image shape of [224, 224, 3] ([height, width, color_channels]).
random_image_size_tensor = torch.rand(size=(224, 224, 3))
print("random_img_siz shape:", random_image_size_tensor.shape,"rad_img_siz_dim:", random_image_size_tensor.ndim)


#creating zero tensor

zero_tensor = torch.zeros(size=(2,4))
print("zero tensor", zero_tensor)

# Can also create a tensor of zeros similar to another tensor
like_zeros_tensor = torch.zeros_like(input=zero_tensor) # will have same shape
like_one_tensor = torch.ones_like(input=zero_tensor)
print(like_one_tensor)

#The most common type (and generally the default) is torch.float32 or torch.float.
#But there's also 16-bit floating point (torch.float16 or torch.half) and 64-bit floating point (torch.float64 or torch.double)

# Default datatype for tensors is float32
float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=None, # defaults to None, which is torch.float32 or whatever datatype is passed
                               device=None, # defaults to None, which uses the default tensor type
                               requires_grad=False) # if True, operations performed on the tensor are recorded 

print(float_32_tensor.shape, float_32_tensor.dtype, float_32_tensor.device)

#Aside from shape issues (tensor shapes don't match up), two of the other most common issues you'll come across in PyTorch are datatype and device issues.

#16bit data type 
float_16_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=torch.float16) # torch.half would also work

print(float_16_tensor.dtype)


#gettting information of tensor 
# Create a tensor
some_tensor = torch.rand(3, 4)

# Find out details about it
print(some_tensor)
print(f"Shape of tensor: {some_tensor.shape}")
print(f"Datatype of tensor: {some_tensor.dtype}")
print(f"Device tensor is stored on: {some_tensor.device}") # will default to CPU

#