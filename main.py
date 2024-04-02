import geopandas as gpd
import numpy as np
pontos = gpd.read_file("/home/lucas/Documentos/projetos/workshop_krig/MetKrig/CAT_Object_pt.shp")

lim = gpd.read_file("/home/lucas/Documentos/projetos/workshop_krig/MetKrig/recorte_pol.shp")

import matplotlib.pyplot as plt

fig,eixo = plt.subplots()

lim.plot(color="blue", ax=eixo)
pontos.plot(column="Avg_z", ax=eixo, legend= True, cmap="YlOrRd")
plt.show()

xmin,ymin,xmax,ymax = pontos.total_bounds

gridx = np.arange(xmin-5, xmax+5, 10, dtype="float64")
gridy = np.arange(ymin-5, ymax+5, 10, dtype="float64")

xi,yi = np.meshgrid(gridx, gridy)

import pykrige.kriging_tools as kt
from pykrige.ok import OrdinaryKriging

x = pontos.geometry.x
y = pontos.geometry.y
z = pontos["Avg_z"]

krig_interp_sph = OrdinaryKriging(x,y,z,
                                  variogram_model="spherical",
                                  verbose=True,
                                  enable_plotting=True,
                                  enable_statistics=True,
                                  nlags=100)


krig_z_sph, krig_sph_var = krig_interp_sph.execute("grid", gridx, gridy)

fig, eixos = plt.subplots(nrows=1,ncols=2, figsize=(12,8))

cax = eixos[0].imshow(krig_z_sph, origin="lower", extent=(xmin, xmax, ymin, ymax),cmap="YlOrRd")
eixos[0].set_title("Interpolação por krigagem com modelo esférico")

pontos.plot(color="blue", ax=eixos[0], label="Amostras", marker="+")

cbarx = plt.colorbar(cax, fraction=0.06)
eixos[0].legend()

cay = eixos[1].imshow(krig_sph_var, origin="lower", extent=(xmin, xmax, ymin, ymax),cmap="RdYlGn_n")
cbary = plt.colorbar(cay, fraction=0.06)

eixos[1].set_title("Variância da interpolação por krigagem com modelo esférico")
eixos[1].legend()

plt.tight_layout()
plt.show()