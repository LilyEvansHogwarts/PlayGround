* msp generated netlist file
.lib '/export/home/svnhome/process/smic.40//hspice/v1p4/l0040ll_v1p4_1r.lib' SS
.lib '/export/home/svnhome/process/smic.40//hspice/v1p4/l0040ll_v1p4_1r.lib' RES_SS

.temp 25

.param lc1 = 6.999999999999999895e-06
.param lc2 = 6.999999999999999895e-06
.param lc3 = 5.000000000000000409e-06
.param vcp = 2.2000000000000001776
.param vdd33 = 3.6299999999999998934
.param wc1 = 6.999999999999999895e-06
.param wc2 = 1.0000000000000000818e-05
.param wc3 = 1.0000000000000000818e-05
.param lb1 = 7.4796668521194868844e-06
.param lb2 = 4.6234984604665243542e-06
.param lb3 = 1.9921439579328586176e-06
.param lb4 = 2.0659234834706269056e-06
.param li10 = 2.5347185763949283685e-06
.param ln4 = 1.9999693637955573257e-06
.param lnsupp = 1.5450562593318438726e-06
.param lnsupp2 = 1.6894925092979854683e-06
.param lp4 = 4.9999999999999987149e-07
.param lpdbin = 7.7409258602206068419e-07
.param lpdin = 1.0022538167561919362e-06
.param lumid = 8.9493537904920256696e-07
.param luumid = 5.9999999999999997285e-07
.param q_lc = 1.9180863970980893781e-06
.param q_llower = 9.000000000000000228e-06
.param q_lq = 1.0000012053547238306e-06
.param q_lref = 3.999999999999998972e-06
.param q_lupper = 1.9981751556863016333e-06
.param q_wc = 1.7204293455814529166e-05
.param q_wlower = 1.1248658643461428382e-06
.param q_wq = 9.957943621581986739e-06
.param q_wref = 1.1136194244203461627e-05
.param q_wupper = 5.2516502665018329854e-06
.param wb1 = 2.5000000000000004586e-05
.param wb2 = 9.9999999999999974299e-07
.param wb3 = 2.6630767393325556938e-06
.param wb4 = 4.4990296354230511014e-06
.param wi10 = 2.2482974054985550616e-06
.param wn4 = 1.7366290694626776833e-06
.param wnsupp = 3.0941699995483912274e-06
.param wnsupp2 = 1.4599453926279649291e-06
.param wp4 = 9.9999999999999974299e-07
.param wpdbin = 1.0598923758660845222e-05
.param wpdin = 3.7452351899862061876e-06
.param wumid = 2.4999999980860549754e-05
.param wuumid = 4.999999999999999562e-06



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


