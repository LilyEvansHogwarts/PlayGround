folder::abs_process_root is "/export/home/svnhome/process/smic.40/";
file::parameter is "charge_pump_param"      # here dcdc_param is the parameter file
{
    param_file_genre = "hspice";
	param q_llower = (4.06266400279797e-06 ~ 1.21879920083939e-05)[ic:8.125328005595933e-06];
	param q_wlower = (8.98873885012438e-07 ~ 2.69662165503731e-06)[ic:1.7977477700248765e-06];
	param q_lupper = (8.1868418310336e-07 ~ 2.45605254931008e-06)[ic:1.6373683662067197e-06];
	param q_wupper = (9.72450669139907e-06 ~ 2.91735200741972e-05)[ic:1.944901338279814e-05];
	param q_lc = (8.99063523927601e-07 ~ 2.6971905717828e-06)[ic:1.7981270478552027e-06];
	param q_wc = (6.91641844792765e-06 ~ 2.0749255343783e-05)[ic:1.3832836895855306e-05];
	param q_lref = (6.02894046560787e-07 ~ 1.80868213968236e-06)[ic:1.2057880931215734e-06];
	param q_wref = (2.88478444434827e-06 ~ 8.65435333304481e-06)[ic:5.769568888696537e-06];
	param q_lq = (1.80161160198273e-06 ~ 5.40483480594818e-06)[ic:3.603223203965456e-06];
	param q_wq = (6.46769426837091e-06 ~ 1.94030828051127e-05)[ic:1.2935388536741826e-05];
	param lpdbin = (2.80240850736784e-07 ~ 8.40722552210351e-07)[ic:5.604817014735675e-07];
	param wpdbin = (5.24687363165373e-06 ~ 1.57406208949612e-05)[ic:1.049374726330746e-05];
	param lpdin = (5.35515754498598e-07 ~ 1.60654726349579e-06)[ic:1.0710315089971964e-06];
	param wpdin = (1.10115773438788e-06 ~ 3.30347320316364e-06)[ic:2.202315468775761e-06];
	param luumid = (3.48972873793311e-07 ~ 1.04691862137993e-06)[ic:6.979457475866211e-07];
	param wuumid = (3.98819853786956e-06 ~ 1.19645956136087e-05)[ic:7.97639707573911e-06];
	param lumid = (3.70478982812236e-07 ~ 1.11143694843671e-06)[ic:7.409579656244713e-07];
	param wumid = (8.35024053134608e-06 ~ 2.50507215940382e-05)[ic:1.6700481062692163e-05];
	param lp4 = (3.51601301600896e-07 ~ 1.05480390480269e-06)[ic:7.032026032017914e-07];
	param wp4 = (1.14010280631222e-06 ~ 3.42030841893665e-06)[ic:2.2802056126244325e-06];
	param ln4 = (4.49328398424184e-07 ~ 1.34798519527255e-06)[ic:8.986567968483671e-07];
	param wn4 = (1.44452671549989e-06 ~ 4.33358014649967e-06)[ic:2.8890534309997816e-06];
	param lnsupp = (4.59256689014719e-07 ~ 1.37777006704416e-06)[ic:9.185133780294376e-07];
	param wnsupp = (1.21099781346365e-06 ~ 3.63299344039096e-06)[ic:2.421995626927309e-06];
	param lnsupp2 = (5.96724073712728e-07 ~ 1.79017222113818e-06)[ic:1.1934481474254553e-06];
	param wnsupp2 = (9.09910783761848e-07 ~ 2.72973235128555e-06)[ic:1.8198215675236967e-06];
	param li10 = (2.10172052674942e-06 ~ 6.30516158024826e-06)[ic:4.203441053498842e-06];
	param wi10 = (9.1022023366048e-07 ~ 2.73066070098144e-06)[ic:1.8204404673209598e-06];
	param lb1 = (2.23941916532887e-06 ~ 6.7182574959866e-06)[ic:4.478838330657734e-06];
	param wb1 = (7.24672826781861e-06 ~ 2.17401848034558e-05)[ic:1.4493456535637218e-05];
	param lb2 = (2.15405461675202e-06 ~ 6.46216385025606e-06)[ic:4.308109233504038e-06];
	param wb2 = (1.31201077333173e-06 ~ 3.9360323199952e-06)[ic:2.624021546663468e-06];
	param lb3 = (7.67115287990454e-07 ~ 2.30134586397136e-06)[ic:1.534230575980909e-06];
	param wb3 = (2.7266068167205e-06 ~ 8.17982045016151e-06)[ic:5.4532136334410055e-06];
	param lb4 = (4.13527226648426e-07 ~ 1.24058167994528e-06)[ic:8.270544532968526e-07];
	param wb4 = (5.73051638169395e-06 ~ 1.71915491450818e-05)[ic:1.14610327633879e-05];
    param lc1 = 7e-6 ;
    param wc1 = 7e-6 ;
    param lc2 = 7e-6 ;
    param wc2 = 10e-6;
    param lc3 = 5e-6 ;
    param wc3 = 10e-6;

    param vcp   = 2.2;

    param vdd33 = 3.3;
    # sweep vdd33 = { 2.97, 3.3, 3.63 };
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

    temperature pll_temp = { "125", "-40" };
    temperature pll_temp_simple = "25";
};

# define options for simulater
file::simulation is "charge_pump"
{
    netlist net
    {
#    sim_corner = TT;
#    sim_temperature = pll_temp_simple;
#    net_proc_rel = "MUL";
        sim_tool        = "hspice64";
        sim_corner      = { TT};
        sim_temperature = pll_temp_simple;

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
