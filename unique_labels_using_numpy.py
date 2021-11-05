import numpy as np
orgin_file =  r'.\000000201706_unique_test.txt'
writed_file =  r'.\000000201706_unique_test_rewrite.txt'
s = np.loadtxt(orgin_file)
print(f"    {s.shape=}")
_, idx = np.unique(s,axis=0, return_index=True)
s_unique = s[np.sort(idx)]
print(s_unique)

print(f"    {s_unique.shape=}")
np.savetxt(writed_file, s_unique,fmt="%g", delimiter=" ")
s_unique_loaded = np.loadtxt(writed_file)
print(f"{    s_unique_loaded.shape=}")
