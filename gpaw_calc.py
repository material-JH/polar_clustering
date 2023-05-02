from gpaw import PW, FermiDirac
from ase.io import read
from gpaw import GPAW

fname = 'XDATCAR_sel'
for n, atoms in enumerate(read(fname, index=':')):
    atoms = read(fname)
    kpts = {'size': (2, 2, 2), 'gamma': True} 
    gpaw = GPAW(mode=PW(300), # plane wave cutoff of 300 eV
                xc='PBE', # exchange-correlation functional
                occupations=FermiDirac(0.1), # Fermi-Dirac smearing with a width of 0.1 eV
                txt='output.txt', kpts=kpts,
                spinpol=False,
                parallel={'domain': 12, 'band': 4}
                ) # file to write output to

    atoms.calc = gpaw
    atoms.get_potential_energy()
    gpaw.write(f'bsto{n}.gpw')