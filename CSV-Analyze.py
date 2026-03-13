import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# 1. LOAD AND CLEAN
df = pd.read_csv('training_log.csv')
df.columns = ['Lambda', 'TV', 'PSNR']

# 2. GROUP BY INPUTS (Average the 'Top' PSNR)
df_avg = df.groupby(['Lambda', 'TV'], as_index=False)['PSNR'].mean()

# 3. COORDINATE TRANSFORMATION
# We use log10 for the axes because Lambda/TV usually vary by orders of magnitude
x = np.log10(df_avg['Lambda'].values)
y = np.log10(df_avg['TV'].values)
z = df_avg['PSNR'].values

# 4. SETUP PLOT
fig = plt.figure(figsize=(14, 9))
ax = fig.add_subplot(111, projection='3d')

# 5. THE "PSEUDO-HEIGHT" SURFACE
# We use 'edgecolor' and 'antialiased' to make the facets visible
surf = ax.plot_trisurf(x, y, z, 
                       cmap='terrain', 
                       edgecolor='black', 
                       linewidth=0.1, 
                       antialiased=True,
                       alpha=0.9)

# 6. FORCE VERTICAL EXAGGERATION (THE FIX)
# This forces the vertical axis to be 3 TIMES TALLER than the width of the base.
# If it's still flat, change 3.0 to 10.0.
ax.set_box_aspect((1, 1, 3.0)) 

# 7. LOCK THE ZOOM (20 to 35)
ax.set_zlim(20, 35)

# 8. RELABEL LOG AXES TO REAL VALUES
ax.set_xlabel('Log10(Lambda)', labelpad=15)
ax.set_ylabel('Log10(TV)', labelpad=15)
ax.set_zlabel('PSNR (dB)', labelpad=15)

# Ensure no scientific notation on Z
ax.zaxis.set_major_formatter(ScalarFormatter(useOffset=False))

plt.title('Compressed Sensing: Optimization Peaks (Faceted Surface)', pad=30)
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

# Set the view to look at the 'profile' of the peaks
ax.view_init(elev=15, azim=135)

plt.show()