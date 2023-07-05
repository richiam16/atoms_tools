import pandas as pd
import matplotlib.pyplot as plt
import re

class Am_Mol:
    """Class used to define an atom, result of a electronic structure calculation""" 
    
    def __init__(self, sum):
        """ The object is initialized just with the path of any .sum file"""
        self.sum = sum

    def _nom(self):
        """Method to get information from the file name of the sum, the file is assumed to have the format: atomOxstate_valence.sum"""
        lisnom = self.sum.split("/")
        lisnom = lisnom[len(lisnom) - 1].split(".")
        self.nom = lisnom[0] # Name of the file
        temp = self.nom.split("_")
        self.tipo = temp[1] # type of calculation, eg alpha, beta, total
        sep = re.split(r'[\+\-]', temp[0])
        if len(sep) <= 1:
            # self.ox = 0
            self.atm = temp[0][0:-1]
            self.ox = temp[0][-1:]
        else:
            self.ox = temp[0][-2:]
            self.atm = sep[0]

    def _rdsum(self):
        """Method to read the information from the .sum file, it creates an individual dataframe called completo with IQA informaiton"""
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
        """Method to include the experimental informaiton of the Atominc Ionization based on the EI_Atoms.txt file"""
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
        """Method to include the experimental informaiton of ELectron Affinity based on the EI_Atoms.txt file"""
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
        """Method to get the atomic number of each atom based on the simbolos_atoms.txt file"""
        for linea in lineas:
            temp = linea.split()
            if temp[2] == self.atm:
                # if temp[0] == self.atm:
                self.atm = temp[2]
                self.z = temp[1]

    def _main(self, lineas, lines, lines_ae):
        self._nom()
        self._az(lineas)
        self._rdsum()
        self._search_EA(lines)
        self._search_AE(lines_ae)

    @staticmethod
    def _search(lines, phrase):
        """ A simple method to find a particular text phrase in a file"""
        for i in range(0, len(lines)):
            a = lines[i].find(phrase)
            if a >= 0:
                lim = i
                break
        return lim

    @staticmethod
    def _crdf(lines, lim):
        """A method to treat simbolos_atoms.txt file"""
        nom_list = lines[lim].split()[2:]
        prop_list = lines[lim + 2].split()[1:]
        df = pd.DataFrame(prop_list, index=nom_list).T
        return df

def iden_op(phrase, name="df"):
    """A method to create a column that is a result of an operation from other columns of a dataframe"""
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
    """ A method to plot columns from dataframe"""
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
