folder::abs_process_root is "/export/home/svnhome/process/smic.40/";
file::parameter is "charge_pump_param"      # here dcdc_param is the parameter file
{
    param_file_genre = "hspice";
	param q_llower = (4.5e-06 ~ 1.35e-05)[ic:9e-06];
	param q_wlower = (5.62432932173071e-07 ~ 1.68729879651921e-06)[ic:1.1248658643461428e-06];
	param q_lupper = (9.99087577843151e-07 ~ 2.99726273352945e-06)[ic:1.9981751556863016e-06];
	param q_wupper = (2.62582513325092e-06 ~ 7.87747539975275e-06)[ic:5.251650266501833e-06];
	param q_lc = (9.59043198549045e-07 ~ 2.87712959564713e-06)[ic:1.9180863970980894e-06];
	param q_wc = (8.60214672790727e-06 ~ 2.58064401837218e-05)[ic:1.720429345581453e-05];
	param q_lref = (2e-06 ~ 6e-06)[ic:4e-06];
	param q_wref = (5.56809712210173e-06 ~ 1.67042913663052e-05)[ic:1.1136194244203462e-05];
	param q_lq = (5.00000602677362e-07 ~ 1.50000180803209e-06)[ic:1.0000012053547238e-06];
	param q_wq = (4.97897181079099e-06 ~ 1.4936915432373e-05)[ic:9.957943621581987e-06];
	param lpdbin = (3.8704629301103e-07 ~ 1.16113887903309e-06)[ic:7.740925860220607e-07];
	param wpdbin = (5.29946187933042e-06 ~ 1.58983856379913e-05)[ic:1.0598923758660845e-05];
	param lpdin = (5.01126908378096e-07 ~ 1.50338072513429e-06)[ic:1.002253816756192e-06];
	param wpdin = (1.8726175949931e-06 ~ 5.61785278497931e-06)[ic:3.745235189986206e-06];
	param luumid = (3e-07 ~ 9e-07)[ic:6e-07];
	param wuumid = (2.5e-06 ~ 7.5e-06)[ic:4.9999999999999996e-06];
	param lumid = (4.47467689524601e-07 ~ 1.3424030685738e-06)[ic:8.949353790492026e-07];
	param wumid = (1.24999999904303e-05 ~ 3.74999999712908e-05)[ic:2.499999998086055e-05];
	param lp4 = (2.5e-07 ~ 7.5e-07)[ic:4.999999999999999e-07];
	param wp4 = (5e-07 ~ 1.5e-06)[ic:9.999999999999997e-07];
	param ln4 = (9.99984681897779e-07 ~ 2.99995404569334e-06)[ic:1.9999693637955573e-06];
	param wn4 = (8.68314534731339e-07 ~ 2.60494360419402e-06)[ic:1.7366290694626777e-06];
	param lnsupp = (7.72528129665922e-07 ~ 2.31758438899777e-06)[ic:1.5450562593318439e-06];
	param wnsupp = (1.5470849997742e-06 ~ 4.64125499932259e-06)[ic:3.0941699995483912e-06];
	param lnsupp2 = (8.44746254648993e-07 ~ 2.53423876394698e-06)[ic:1.6894925092979855e-06];
	param wnsupp2 = (7.29972696313982e-07 ~ 2.18991808894195e-06)[ic:1.459945392627965e-06];
	param li10 = (1.26735928819746e-06 ~ 3.80207786459239e-06)[ic:2.534718576394928e-06];
	param wi10 = (1.12414870274928e-06 ~ 3.37244610824783e-06)[ic:2.248297405498555e-06];
	param lb1 = (3.73983342605974e-06 ~ 1.12195002781792e-05)[ic:7.479666852119487e-06];
	param wb1 = (1.25e-05 ~ 3.75e-05)[ic:2.5e-05];
	param lb2 = (2.31174923023326e-06 ~ 6.93524769069979e-06)[ic:4.623498460466524e-06];
	param wb2 = (5e-07 ~ 1.5e-06)[ic:9.999999999999997e-07];
	param lb3 = (9.96071978966429e-07 ~ 2.98821593689929e-06)[ic:1.9921439579328586e-06];
	param wb3 = (1.33153836966628e-06 ~ 3.99461510899883e-06)[ic:2.6630767393325557e-06];
	param lb4 = (1.03296174173531e-06 ~ 3.09888522520594e-06)[ic:2.065923483470627e-06];
	param wb4 = (2.24951481771153e-06 ~ 6.74854445313458e-06)[ic:4.499029635423051e-06];
    param lc1 = 7e-6 ;
    param wc1 = 7e-6 ;
    param lc2 = 7e-6 ;
    param wc2 = 10e-6;
    param lc3 = 5e-6 ;
    param wc3 = 10e-6;

#  param vdd33 = 3.3;
    param vcp   = 2.2;

    sweep vdd33 = { 2.97, 3.3, 3.63 };
#  sweep vcp = { 1.0, 1.7, 2.5 };
};

file::process is "charge_pump_process"
{
    corner { SS, FF, TT }
    {
        lib "/hspice/v1p4/l0040ll_v1p4_1r.lib" "$$";
        lib "/hspice/v1p4/l0040ll_v1p4_1r.lib" "RES_$$";
    };

    corner { SNFP, FNSP }
    {
        lib "/hspice/v1p4/l0040ll_v1p4_1r.lib" "$$";
        lib "/hspice/v1p4/l0040ll_v1p4_1r.lib" "RES_TT";
    };

    temperature pll_temp = { "125", "-40", "25" };
    temperature pll_temp_simple = "25";
};

# define options for simulater
file::simulation is "charge_pump"
{
    netlist net
    {
        sim_tool = "hspice64";
#    sim_corner = TT;
#    sim_temperature = pll_temp_simple;
        sim_corner = { TT, FF, SS };
        sim_temperature = pll_temp;
#    net_proc_rel = "MUL";

        net_source  = "tran.sp";
        sim_depend  = "netlist";

        extract { "up_imin", "up_iavg", "up_imax" } from "tran.mt0";
        extract { "lo_imin", "lo_iavg", "lo_imax" } from "tran.mt0";
    };
};

operation::specification is "charge_pump"
{
    const std_i   = 40;   # uA
    var up_imin   = net::up_imin * 1e6; # turn to uA
    var up_iavg   = net::up_iavg * 1e6; # turn to uA
    var up_imax   = net::up_imax * 1e6; # turn to uA
    var lo_imin   = net::lo_imin * 1e6; # turn to uA
    var lo_iavg   = net::lo_iavg * 1e6; # turn to uA
    var lo_imax   = net::lo_imax * 1e6; # turn to uA
    var diff_upup = max( up_imax - up_iavg );
    var diff_updn = max( up_iavg - up_imin );
    var diff_loup = max( lo_imax - lo_iavg );
    var diff_lodn = max( lo_iavg - lo_imin );
    var sum_diff  = diff_upup + diff_updn + diff_loup + diff_lodn;
    var deviation = max( abs( up_iavg - std_i ) ) + max( abs( lo_iavg - std_i ) );
    var target    = sum_diff * 0.3 + deviation * 0.5;
    fom target;

    probe { up_imin, up_iavg, up_imax };
    probe { lo_imin, lo_iavg, lo_imax };
    probe { diff_upup, diff_updn, diff_loup, diff_lodn };
    probe { sum_diff, deviation };

    constraint diff_upup < 20;
    constraint diff_updn < 20;
    constraint diff_loup < 5;
    constraint diff_lodn < 5;
    constraint deviation < 5;
};

operation::optimization is "charge_pump_opt"
{
    opt_max_iter   = 1;
    opt_max_solver = 1;
    opt_option     = "0";
    opt_strategy   = "TC";
    opt_algo       = "SLSQP";
    opt_stat       = "pf";
};
