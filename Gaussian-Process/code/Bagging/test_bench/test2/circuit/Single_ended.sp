*Netlist
*Operational Amplifier with Single-Ended Output
*Author: Minghua Li
*University of Texas at Dallas
*Email: mxl095420@utdallas.edu
*9/23/2012


.lib './tsmc18/rf018.l' TT_3V

.inc 'param'

.param vdd_v    = 2
.param vin_cm   = '0.5*vdd_v'
.param vin_low  = '0.375*vdd_v'
.param vin_high = '0.625*vdd_v'
.param risetime = 1e-9
.param falltime = 1e-9
.param period   = 100e-6

.option post
.option ingold=2
.option numdgt=10
*.option seed='random' randgen='moa'

.inc netlist 

xac     vin_ac+     vin_ac-     vo_ac   vdd_ac      vss opamp
xtr     vin_tr+     vo_tr       vo_tr   vdd_tr      vss opamp
xsw     vin_sw+     vin_sw-     vo_sw   vdd_sw      vss opamp
xpsr    vin_psr+    vo_psr      vo_psr  vdd_psrac   vss opamp
xcmrr   vin_cmrr+   vin_cmrr-   vo_cmrr vdd_cmrr    vss opamp


rsw10   vin_sw-     vo_sw       10meg
rsw1    vin_sw      vin_sw-     1meg
vin_sw  vin_sw      0           vin_cm
vin_sw+ vin_sw+     0           vin_cm

vin_psr+    vin_psr+    0       vin_cm

e_cmrr+     vin_cmrr+   vcm_cmrr    100cmrr     0           0.5
e_cmrr-     vin_cmrr-   vo_cmrr     100cmrr     0           0.5
vcm_cmrr    vcm_cmrr    0           dc=vin_cm
vs_cmrr     100cmrr     0           dc=0        ac=1


vdd_ac      vdd_ac      0       vdd_v
vdd_tr      vdd_tr      0       vdd_v
vdd_sw      vdd_sw      0       vdd_v
vdd_psr     vdd_psr     0       vdd_v
vdd_psrac   vdd_psrac   vdd_psr dc=0    ac=1
vdd_cmrr    vcc_cmrr    0       vdd_v
vss         vss         0       0

e+      vin_ac+     101     100     0 0.5
e-      vin_ac-     101     100     0 -0.5
vcm     101         0       dc=vin_cm
vs      100         0       dc=0    ac=1


.ac   dec 100 1 50g

.meas ac  gain max   vdb(vo_ac)   from=1 to=50g
.meas ac  ugf  when vdb(vo_ac)=0
.meas ac  phase min  vp(vo_ac) from=1 to=ugf
.meas ac  pm=param('phase + 180')
* .meas ac  gm   find  vdb(vo_ac)   when   vp(vo_ac)=-178
* .meas ac  psr  min   vdb(vo_psr)  from=1 to=50g
* .meas ac  cmrr min   vdb(vo_cmrr) from=1 to=50g

.end
