import torch

tensor = torch.tensor([1,2,3,4])
print(tensor + 10)
#output tensor([2, 3, 4, 5])

print(tensor * 10)
#output tensor([10, 20, 30, 40])

#Notice how the tensor values above didn't end up being tensor([110, 120, 130]), this is because the values inside the tensor don't change unless they're reassigned.

print(tensor)
#output tensor([1,2,3,4])

tensor -= 10
print(tensor)
#output tensor([-9, -8, -7, -6])

tensor += 11
print(tensor)
#output tensor([2, 3, 4, 5])

torch.multiply(tensor,2)
print(tensor)
# Original tensor is still unchanged 

# Element-wise multiplication (each element multiplies its equivalent, index 0->0, 1->1, 2->2)
print(tensor, "*", tensor)
print("Equals:", tensor * tensor)

#metrix multiplication
tensor = torch.tensor([1, 2, 3])
print(tensor.matmul(tensor))
print(tensor @ tensor)


# transposal of tensor 

# Shapes need to be in the right way  
tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]], dtype=torch.float32)

tensor_B = torch.tensor([[7, 10],
                         [8, 11], 
                         [9, 12]], dtype=torch.float32)

# View tensor_A and tensor_B
print(tensor_A)
print(tensor_B)

# View tensor_A and tensor_B.T
print(tensor_A)
print(tensor_B.T)

#this make b tensor colum match with row of  tensor its just swap the demension like (3,2) to (2,3)

#torch.transpose(input, dim0, dim1) - where input is the desired tensor to transpose and dim0 and dim1 are the dimensions to be swapped.
#tensor.T - where tensor is the desired tensor to transpose.

# Create a tensor
tensor = torch.arange(10, 100, 10)
print(f"Tensor: {tensor}")

# Returns index of max and min values
print(f"Index where max value occurs: {tensor.argmax()}")
print(f"Index where min value occurs: {tensor.argmin()}")

#changing datatype of tensor
print(tensor.dtype)

new_tensor = tensor.type(torch.float16)
print(new_tensor.dtype)
print(new_tensor)



