* msp generated netlist file
.lib '/export/home/svnhome/process/smic.40//hspice/v1p4/l0040ll_v1p4_1r.lib' FF
.lib '/export/home/svnhome/process/smic.40//hspice/v1p4/l0040ll_v1p4_1r.lib' RES_FF

.temp -40

.param lc1 = 6.999999999999999895e-06
.param lc2 = 6.999999999999999895e-06
.param lc3 = 5.000000000000000409e-06
.param vcp = 2.2000000000000001776
.param vdd33 = 3.6299999999999998934
.param wc1 = 6.999999999999999895e-06
.param wc2 = 1.0000000000000000818e-05
.param wc3 = 1.0000000000000000818e-05
.param lb1 = 7.4463986148460239485e-06
.param lb2 = 6.0208415163314612042e-06
.param lb3 = 1.935205274807262194e-06
.param lb4 = 1.796776990644936034e-06
.param li10 = 2.1281518946776708235e-06
.param ln4 = 1.235724785222264301e-06
.param lnsupp = 1.4895854097497889148e-06
.param lnsupp2 = 1.519907787678445467e-06
.param lp4 = 1.6212767428764992747e-06
.param lpdbin = 1.6587851635049963541e-06
.param lpdin = 5.5276659511929450804e-07
.param lumid = 1.3962872366816469556e-06
.param luumid = 1.1784285402359399645e-06
.param q_lc = 1.8985702602905687531e-06
.param q_llower = 7.5678154022788825895e-06
.param q_lq = 3.5078976954945712235e-06
.param q_lref = 1.4656271765991558511e-06
.param q_lupper = 1.4519778237694768636e-06
.param q_wc = 7.9712192447025629504e-06
.param q_wlower = 2.2733336454883651862e-06
.param q_wq = 1.1038705723344060107e-05
.param q_wref = 1.78009213299980139e-05
.param q_wupper = 8.1694945060286369796e-06
.param wb1 = 1.4140242828755245579e-05
.param wb2 = 1.3842257868385174748e-06
.param wb3 = 1.4548072124072563394e-06
.param wb4 = 8.7148924556304877416e-06
.param wi10 = 1.7609560790400517192e-06
.param wn4 = 1.6638058815366056502e-06
.param wnsupp = 1.711788789492475284e-06
.param wnsupp2 = 2.8896950165582203029e-06
.param wp4 = 1.1980216175725355816e-06
.param wpdbin = 7.6009723777729275252e-06
.param wpdin = 5.0095740023895392256e-06
.param wumid = 1.9996991320842548749e-05
.param wuumid = 1.8936433710324601838e-05



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


