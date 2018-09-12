folder::prj_dir is "/export/home/wllv/shzhang/Bagging/test_bench/test3/circuit";
folder::abs_process_root is "/export/home/svnhome/process/smic.40/";
file::parameter is "charge_pump_param"      # here dcdc_param is the parameter file
{
    param_file_genre = "hspice";
	param q_llower = (4.40314713316879e-06 ~ 1.32094413995064e-05)[ic:8.806294266337578e-06];
	param q_wlower = (8.59277301592327e-07 ~ 2.57783190477698e-06)[ic:1.7185546031846546e-06];
	param q_lupper = (7.18983005781942e-07 ~ 2.15694901734582e-06)[ic:1.437966011563883e-06];
	param q_wupper = (7.29028645784206e-06 ~ 2.18708593735262e-05)[ic:1.4580572915684128e-05];
	param q_lc = (8.81898121224105e-07 ~ 2.64569436367232e-06)[ic:1.7637962424482105e-06];
	param q_wc = (8.46911547522588e-06 ~ 2.54073464256776e-05)[ic:1.6938230950451762e-05];
	param q_lref = (1.34065667227103e-06 ~ 4.02197001681308e-06)[ic:2.6813133445420513e-06];
	param q_wref = (4.7321882928568e-06 ~ 1.41965648785704e-05)[ic:9.464376585713598e-06];
	param q_lq = (6.93427461546856e-07 ~ 2.08028238464057e-06)[ic:1.3868549230937124e-06];
	param q_wq = (5.15331638757854e-06 ~ 1.54599491627356e-05)[ic:1.0306632775157075e-05];
	param lpdbin = (4.55026544079624e-07 ~ 1.36507963223887e-06)[ic:9.100530881592474e-07];
	param wpdbin = (6.39235727470115e-06 ~ 1.91770718241035e-05)[ic:1.2784714549402305e-05];
	param lpdin = (3.05504632752411e-07 ~ 9.16513898257234e-07)[ic:6.110092655048224e-07];
	param wpdin = (2.41461091914351e-06 ~ 7.24383275743054e-06)[ic:4.829221838287027e-06];
	param luumid = (4.96736164587413e-07 ~ 1.49020849376224e-06)[ic:9.934723291748255e-07];
	param wuumid = (9.19724738865889e-06 ~ 2.75917421659767e-05)[ic:1.8394494777317767e-05];
	param lumid = (4.13843035403857e-07 ~ 1.24152910621157e-06)[ic:8.276860708077141e-07];
	param wumid = (6.38828827295503e-06 ~ 1.91648648188651e-05)[ic:1.2776576545910061e-05];
	param lp4 = (7.75986483232811e-07 ~ 2.32795944969843e-06)[ic:1.5519729664656213e-06];
	param wp4 = (6.02509713863907e-07 ~ 1.80752914159172e-06)[ic:1.2050194277278145e-06];
	param ln4 = (5.25225077938878e-07 ~ 1.57567523381663e-06)[ic:1.0504501558777564e-06];
	param wn4 = (1.01662177183059e-06 ~ 3.04986531549178e-06)[ic:2.0332435436611846e-06];
	param lnsupp = (5.72598994381817e-07 ~ 1.71779698314545e-06)[ic:1.1451979887636345e-06];
	param wnsupp = (1.3272763771489e-06 ~ 3.98182913144671e-06)[ic:2.654552754297807e-06];
	param lnsupp2 = (7.48367180225511e-07 ~ 2.24510154067653e-06)[ic:1.4967343604510215e-06];
	param wnsupp2 = (9.10282242373514e-07 ~ 2.73084672712054e-06)[ic:1.8205644847470283e-06];
	param li10 = (1.34425291310184e-06 ~ 4.03275873930552e-06)[ic:2.6885058262036825e-06];
	param wi10 = (1.26544466223579e-06 ~ 3.79633398670737e-06)[ic:2.5308893244715793e-06];
	param lb1 = (3.69246285806264e-06 ~ 1.10773885741879e-05)[ic:7.384925716125287e-06];
	param wb1 = (4.96139652217307e-06 ~ 1.48841895665192e-05)[ic:9.922793044346136e-06];
	param lb2 = (3.10866158288051e-06 ~ 9.32598474864152e-06)[ic:6.217323165761014e-06];
	param wb2 = (7.05636961628973e-07 ~ 2.11691088488692e-06)[ic:1.411273923257947e-06];
	param lb3 = (2.71696287652143e-07 ~ 8.1508886295643e-07)[ic:5.433925753042868e-07];
	param wb3 = (1.25288176815958e-06 ~ 3.75864530447874e-06)[ic:2.5057635363191583e-06];
	param lb4 = (7.80313134935067e-07 ~ 2.3409394048052e-06)[ic:1.560626269870135e-06];
	param wb4 = (4.74179861995378e-06 ~ 1.42253958598613e-05)[ic:9.48359723990755e-06];
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

    temperature pll_temp = { "125", "-40" };
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
