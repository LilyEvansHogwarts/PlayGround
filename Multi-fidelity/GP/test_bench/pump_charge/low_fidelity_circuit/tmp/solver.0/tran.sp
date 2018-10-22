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
.param lb1 = 4.478838330657733921e-06
.param lb2 = 4.3081092335040382037e-06
.param lb3 = 1.5342305759809089554e-06
.param lb4 = 8.2705445329685258706e-07
.param li10 = 4.2034410534988422796e-06
.param ln4 = 8.9865679684836718225e-07
.param lnsupp = 9.1851337802943758e-07
.param lnsupp2 = 1.1934481474254554754e-06
.param lp4 = 7.0320260320179153386e-07
.param lpdbin = 5.6048170147356752117e-07
.param lpdin = 1.0710315089971961593e-06
.param lumid = 7.4095796562447129694e-07
.param luumid = 6.9794574758662108876e-07
.param q_lc = 1.7981270478552027422e-06
.param q_llower = 8.1253280055959332023e-06
.param q_lq = 3.6032232039654558489e-06
.param q_lref = 1.2057880931215732265e-06
.param q_lupper = 1.6373683662067196667e-06
.param q_wc = 1.383283689585530554e-05
.param q_wlower = 1.7977477700248765353e-06
.param q_wq = 1.2935388536741826144e-05
.param q_wref = 5.7695688886965367951e-06
.param q_wupper = 1.944901338279814146e-05
.param wb1 = 1.4493456535637217597e-05
.param wb2 = 2.6240215466634680009e-06
.param wb3 = 5.4532136334410055353e-06
.param wb4 = 1.1461032763387899998e-05
.param wi10 = 1.8204404673209597503e-06
.param wn4 = 2.8890534309997816285e-06
.param wnsupp = 2.4219956269273090897e-06
.param wnsupp2 = 1.8198215675236966676e-06
.param wp4 = 2.2802056126244325022e-06
.param wpdbin = 1.0493747263307461981e-05
.param wpdin = 2.2023154687757609096e-06
.param wumid = 1.6700481062692162861e-05
.param wuumid = 7.9763970757391100042e-06



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


