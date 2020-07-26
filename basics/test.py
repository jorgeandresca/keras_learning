import site
temp = site.getsitepackages() # List of global package locations

temp2 = site.getusersitepackages() # String for user-specific package location


print(temp)
print(temp2)