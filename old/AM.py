import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import re

arch = []
for filename in glob.glob(os.path.join("C:\\Users\\richi\\Desktop\\QTAIM\\NPA\\nuevos", "*.sum")):
    arch.append(filename)
file = open("simbolos_atomos.txt", "r")
lineas = file.readlines()
file_2 = open("EI_Atoms.txt", "r")
lines = file_2.readlines()
file_3 = open("AE_Atoms.txt", "r")
lines_ae = file_3.readlines()


class Am_Mol:
    def __init__(self, sum):
        self.sum = sum

    def _nom(self):
        lisnom = self.sum.split("\\")
        lisnom = lisnom[len(lisnom) - 1].split(".")
        self.nom = lisnom[0]
        temp = self.nom.split("_")
        self.tipo = temp[1]
        sep = re.split(r'[\+\-]', temp[0])
        if len(sep) <= 1:
            # self.ox = 0
            self.atm = temp[0][0:-1]
            self.ox = temp[0][-1:]
        else:
            self.ox = temp[0][-2:]
            self.atm = sep[0]

    def _rdsum(self):
        print("reading file:" + self.nom)
        file = open(self.sum, "r")
        lines = file.readlines()
        lim = self._search(lines, "Atom A     E_IQA_Intra(A)")
        self.intra = self._crdf(lines, lim)
        if self.tipo == "ab":
            lim_2 = self._search(lines, "Atomic Electronic Spin Populations")
            self.pop = self._crdf(lines, lim_2 + 7)
            self.completo = pd.concat([self.intra, self.pop], axis=1)
        else:
            self.completo = self.intra
        self.completo.insert(0, "Atom", self.atm)
        self.completo.insert(1, "Z", self.z)
        self.completo.insert(1, "Ox", self.ox)
        self.completo.insert(2, "Type", self.tipo)
        self.completo = self.completo.apply(pd.to_numeric, errors="ignore")

    def _search_EA(self, lines):
        if self.tipo == "ab":
            for i in range(3, len(lines) - 1):
                comp = lines[i].split("|")
                atm = comp[1].split()[0]
                ox = float(comp[2])
                if atm == self.atm and ox == float(self.ox):
                    try:
                        self.completo["IE"] = float(comp[8]) / 27.2113845
                    except ValueError:
                        self.completo["IE"] = float(re.findall(
                            r'[0-9\.]+', comp[8])[0]) / 27.2113845
                    break

    def _search_AE(self, lines):
        if self.tipo == "ab":
            for i in range(2, len(lines) - 1):
                comp = lines[i].split("|")
                if comp[1].split()[0] == self.atm and float(comp[2]) == float(self.ox):
                    try:
                        self.completo["EA"] = float(comp[6]) / 27.2113845
                    except ValueError:
                        self.completo["EA"] = float(re.findall(
                            r'[0-9\.]+', comp[6])[0]) / 27.2113845
                    break

    def _az(self, lineas):
        for linea in lineas:
            temp = linea.split()
            if temp[2] == self.atm:
                # if temp[0] == self.atm:
                self.atm = temp[2]
                self.z = temp[1]

    @staticmethod
    def _search(lines, phrase):
        for i in range(0, len(lines)):
            a = lines[i].find(phrase)
            if a >= 0:
                lim = i
                break
        return lim

    @staticmethod
    def _crdf(lines, lim):
        nom_list = lines[lim].split()[2:]
        prop_list = lines[lim + 2].split()[1:]
        df = pd.DataFrame(prop_list, index=nom_list).T
        return df


mol = [Am_Mol(dic) for dic in arch]
for i in range(0, len(mol)):
    mol[i]._nom()
    mol[i]._az(lineas)
    mol[i]._rdsum()
    mol[i]._search_EA(lines)
    mol[i]._search_AE(lines_ae)
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
# master["(E_IQA_Intra(A)_a+E_IQA_Intra(A)_b)/Vee_ab(A,A)"] = (master["E_IQA_Intra(A)_a"] +
#                                                            master["E_IQA_Intra(A)_b"]) / master["Vee_ab(A,A)"]
master = master.sort_values(by=["Z", "Ox"], ascending=True)


def iden_op(phrase, name="df"):
    ope = re.split(r'[\+\-*/]', phrase)
    simb = re.findall(r'[\+\-*/]', phrase)
    new_c = ""
    while len(simb) != len(ope):
        simb.append("")
    for part, simbol in zip(ope, simb):
        try:
            float(part)
            new_c += f"{part}{simbol}"
        except ValueError:
            new_c += f"{name}['{part}']{simbol}"
    return new_c


def graph_a(df, ax_x, ax_y, title="", column="Atom", simb_log="!=", A="A"):
    plt.rcParams.update({'font.size': 15})
    ax = plt.gca()
    df[str(ax_x)] = eval(iden_op(ax_x, "df"))
    try:
        float(A)
        cond = f"df[df['{column}']{simb_log}{A}]"
    except ValueError:
        cond = f"df[df['{column}']{simb_log}'{A}']"
    df = eval(cond)
    for carac in ax_y:
        try:
            df.insert(3, carac, eval(iden_op(carac, "df")))
        except ValueError:
            df = df
        df = df.sort_values(by=[ax_x], ascending=True)
        df.plot(kind="scatter", x=ax_x, y=carac, color="red",
                ax=ax, title=title, fontsize=14)
        df.plot(kind="line", x=ax_x, y=carac, ax=ax, fontsize=14)
    ax.set_ylabel("E(A)")
    # plt.show()
