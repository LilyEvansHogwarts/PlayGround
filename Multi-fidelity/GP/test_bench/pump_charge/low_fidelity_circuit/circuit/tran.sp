***simulation file***
*.option
*+ fast
*+ post node list
*+ method=gear
*+ runlvl=6
*+ probe=1
*+ accurate=6
*+ dcon=-1
*+ modmonte=1

.inc 'netlist'

.ic v(vcp_net)=vcp

.op
.tran 2p 200n $ sweep monte=100

*.probe v(OUT_0) V(LOCK) v(XI0.CK_REF) V(XI0.net0134) v(xi0.net68)  v(xi0.DIV10)  v(xi0.UP)  v(xi0.UPB) v(xi0.DNB) v(xi0.DN)  v(xi0.DN12)  v(xi0.UP12) V(xi0.CP_OUT) V(xi0.vctr) V(xi0.xi66.quench) v(xi0.pd_clk_ready) v(xi0.LOCK_REF) v(xi0.xi60.ph_3) v(xi0.xi60.ph_7) v(xi0.xi60.vcovdd)  v(xi0.xi60.reset) v(xi0.xi60.resetn) 

*.print i(xi0.vupper) i(xi0.vlower)

.measure tran up_imin min i(xi0.vupper) from=20e-9 to=100e-9
.measure tran up_iavg avg i(xi0.vupper) from=20e-9 to=180e-9
.measure tran up_imax max i(xi0.vupper) from=20e-9 to=100e-9

.measure tran lo_imin min i(xi0.vlower) from=20e-9 to=100e-9
.measure tran lo_iavg avg i(xi0.vlower) from=20e-9 to=180e-9
.measure tran lo_imax max i(xi0.vlower) from=20e-9 to=100e-9

.end

