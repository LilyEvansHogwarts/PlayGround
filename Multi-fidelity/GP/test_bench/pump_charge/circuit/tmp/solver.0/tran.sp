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
.param lb1 = 6.4581458326471173669e-06
.param lb2 = 7.0205076298639020244e-06
.param lb3 = 8.0209420104768684006e-07
.param lb4 = 2.71731913428784463e-06
.param li10 = 5.6338533173759847188e-06
.param ln4 = 1.3131922765934229376e-06
.param lnsupp = 1.9498946806019565984e-06
.param lnsupp2 = 1.9269996080895650432e-06
.param lp4 = 1.2305938353148877786e-06
.param lpdbin = 1.1754129409569538279e-06
.param lpdin = 8.8836010120560733938e-07
.param lumid = 9.9562824156311078635e-07
.param luumid = 9.6906151437453500295e-07
.param q_lc = 2.9747733454030488349e-06
.param q_llower = 3.1996513645004408418e-06
.param q_lq = 1.1071366131962694935e-06
.param q_lref = 1.5752475860779397418e-06
.param q_lupper = 1.7933110750422049957e-06
.param q_wc = 9.0570554195453852758e-06
.param q_wlower = 2.0903950100886142925e-06
.param q_wq = 1.2607335706964690176e-05
.param q_wref = 5.7717725797431868189e-06
.param q_wupper = 6.0637392049576812627e-06
.param wb1 = 1.3183396214591633759e-05
.param wb2 = 2.5906047779592056139e-06
.param wb3 = 1.4459809723248085616e-06
.param wb4 = 6.0488369862470741352e-06
.param wi10 = 2.9852695187190104767e-06
.param wn4 = 3.2723341452385715694e-06
.param wnsupp = 3.9325600786937941743e-06
.param wnsupp2 = 3.0300357553126834827e-06
.param wp4 = 1.3408963346079309904e-06
.param wpdbin = 9.8003702176595795489e-06
.param wpdin = 5.0963597499286865251e-06
.param wumid = 1.2376866702923901769e-05
.param wuumid = 7.0636107633852591596e-06



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


