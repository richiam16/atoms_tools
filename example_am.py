from AM import Am_Mol
import glob
import os
import pandas as pd

file = open("df_construction/simbolos_atomos.txt", "r")
lineas = file.readlines()
file_2 = open("df_construction/EI_Atoms.txt", "r")
lines = file_2.readlines()
file_3 = open("df_construction/AE_Atoms.txt", "r")
lines_ae = file_3.readlines()

arch = []
#for filename in glob.glob(os.path.join("C:\\Users\\richi\\Desktop\\QTAIM\\NPA\\nuevos", "*.sum")):
for filename in glob.glob(os.path.join("examples/","*.sum")):
    arch.append(filename)

mol = [Am_Mol(dic) for dic in arch]
for i in range(0, len(mol)):
    mol[i]._main(lineas, lines, lines_ae)
    print("file" + " " + str(i + 1) + " of" + " " + str(len(mol)))
alpha = pd.DataFrame()
beta = pd.DataFrame()
total = pd.DataFrame()
print("creating alpha,beta and total dataframes")
for i in range(0, len(mol)):
    if mol[i].tipo == "a":
        alpha = alpha.append(mol[i].completo, ignore_index=True)
    elif mol[i].tipo == "b":
        beta = beta.append(mol[i].completo, ignore_index=True)
    elif mol[i].tipo == "ab":
        total = total.append(mol[i].completo, ignore_index=True)
alpha_beta = alpha.join(beta.set_index(["Atom", "Ox", "Z"]), on=[
                       "Atom", "Ox", "Z"], lsuffix="_a", rsuffix="_b")

print("creating master dataframe")
master = alpha_beta.join(total.set_index(["Atom", "Ox", "Z"]), on=[
    "Atom", "Ox", "Z"], rsuffix="_t")
master = master.drop(["Type", "Type_a", "Type_b"], axis=1)
a = master["Z"] - master["Ox"]
master.insert(3, "#e", a)
master["Vee_ab(A,A)"] = master["E_IQA_Intra(A)"] - \
    (master["E_IQA_Intra(A)_a"] + master["E_IQA_Intra(A)_b"])
master["(E_IQA_Intra(A)_a+E_IQA_Intra(A)_b)/Vee_ab(A,A)"] = (master["E_IQA_Intra(A)_a"] +
                                                            master["E_IQA_Intra(A)_b"]) / master["Vee_ab(A,A)"]
master = master.sort_values(by=["Z", "Ox"], ascending=True)