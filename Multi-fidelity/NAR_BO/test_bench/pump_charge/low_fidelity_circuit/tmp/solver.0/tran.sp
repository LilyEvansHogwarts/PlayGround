* msp generated netlist file
.lib '/export/home/svnhome/process/smic.40//hspice/v1p4/l0040ll_v1p4_1r.lib' TT
.lib '/export/home/svnhome/process/smic.40//hspice/v1p4/l0040ll_v1p4_1r.lib' RES_TT

.temp 25

.param lc1 = 6.999999999999999895e-06
.param lc2 = 6.999999999999999895e-06
.param lc3 = 5.000000000000000409e-06
.param vcp = 2.2000000000000001776
.param vdd33 = 3.2999999999999998224
.param wc1 = 6.999999999999999895e-06
.param wc2 = 1.0000000000000000818e-05
.param wc3 = 1.0000000000000000818e-05
.param lb1 = 7.7921112776297864048e-06
.param lb2 = 2.0000021123523465953e-06
.param lb3 = 1.1831129935178557893e-06
.param lb4 = 3.000000000000000076e-06
.param li10 = 2.0000000036480365586e-06
.param ln4 = 1.999999999999999486e-06
.param lnsupp = 1.999999999999999486e-06
.param lnsupp2 = 2.3999822985290227378e-06
.param lp4 = 1.2133158616387269344e-06
.param lpdbin = 1.950310665079662842e-06
.param lpdin = 7.0851566882024897444e-07
.param lumid = 5.5000000000000013393e-07
.param luumid = 5.9999999999999997285e-07
.param q_lc = 3.7204111247574521202e-06
.param q_llower = 3.000000000000000076e-06
.param q_lq = 1.0000000000045400513e-06
.param q_lref = 3.999999999999998972e-06
.param q_lupper = 1.999999999999999486e-06
.param q_wc = 1.6968925919616899143e-05
.param q_wlower = 1.1269371272428141296e-06
.param q_wq = 9.973057988255504871e-06
.param q_wref = 5.0000003862141840426e-06
.param q_wupper = 1.1465887791092803534e-05
.param wb1 = 2.5000000000000004586e-05
.param wb2 = 1.0001096190368698828e-06
.param wb3 = 1.0066383748140561522e-06
.param wb4 = 4.0000000023356855477e-06
.param wi10 = 2.24581284254775837e-06
.param wn4 = 1.0000000913194938069e-06
.param wnsupp = 1.0000091092946680085e-06
.param wnsupp2 = 3.1333115723118438371e-06
.param wp4 = 1.0000115202403009449e-06
.param wpdbin = 5.7833424996302961385e-06
.param wpdin = 2.0000000285528097113e-06
.param wumid = 8.0000007803882167254e-06
.param wuumid = 5.0000000014636098144e-06



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


