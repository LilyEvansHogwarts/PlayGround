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
.param lb1 = 4.9411821681527250382e-06
.param lb2 = 5.8303554556079646687e-06
.param lb3 = 1.9135329524622187564e-06
.param lb4 = 2.9883790271213360866e-06
.param li10 = 7.2491611611688602725e-06
.param ln4 = 1.9542278738052780977e-06
.param lnsupp = 1.7887196010873106481e-06
.param lnsupp2 = 1.8753488428736158082e-06
.param lp4 = 1.5306771395453538949e-06
.param lpdbin = 1.152227970353471279e-06
.param lpdin = 9.8945527868652315933e-07
.param lumid = 1.107276344310686734e-06
.param luumid = 9.0132128855538879699e-07
.param q_lc = 3.4305837738037081979e-06
.param q_llower = 8.8736856353425459874e-06
.param q_lq = 2.9441748406446758364e-06
.param q_lref = 3.9809191197581968483e-06
.param q_lupper = 1.4929224286048852562e-06
.param q_wc = 8.7275502236942955925e-06
.param q_wlower = 3.2091165188545807528e-06
.param q_wq = 5.3949284152211002258e-06
.param q_wref = 1.0385874474509439815e-05
.param q_wupper = 5.1235836785927441658e-06
.param wb1 = 1.9218489389935871709e-05
.param wb2 = 1.1089624581012124645e-06
.param wb3 = 1.7601796634874000405e-06
.param wb4 = 1.2705219846810896108e-05
.param wi10 = 1.923967911255295606e-06
.param wn4 = 3.5153456250154412524e-06
.param wnsupp = 2.614224367381223991e-06
.param wnsupp2 = 2.2341347394043318591e-06
.param wp4 = 3.8849771543986679118e-06
.param wpdbin = 4.4688795067920669538e-06
.param wpdin = 2.7794819629220870822e-06
.param wumid = 1.6366144336510448092e-05
.param wuumid = 1.4856455542215096246e-05



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


