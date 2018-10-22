folder::abs_process_root is "/export/home/svnhome/process/smic.40/";
file::parameter is "charge_pump_param"      # here dcdc_param is the parameter file
{
    param_file_genre = "hspice";
	param q_llower = (1.59982568225022e-06 ~ 4.79947704675066e-06)[ic:3.1996513645004404e-06];
	param q_wlower = (1.04519750504431e-06 ~ 3.13559251513292e-06)[ic:2.0903950100886143e-06];
	param q_lupper = (8.96655537521102e-07 ~ 2.68996661256331e-06)[ic:1.793311075042205e-06];
	param q_wupper = (3.03186960247884e-06 ~ 9.09560880743652e-06)[ic:6.06373920495768e-06];
	param q_lc = (1.48738667270152e-06 ~ 4.46216001810457e-06)[ic:2.9747733454030484e-06];
	param q_wc = (4.52852770977269e-06 ~ 1.35855831293181e-05)[ic:9.057055419545385e-06];
	param q_lref = (7.8762379303897e-07 ~ 2.36287137911691e-06)[ic:1.5752475860779397e-06];
	param q_wref = (2.88588628987159e-06 ~ 8.65765886961478e-06)[ic:5.771772579743187e-06];
	param q_lq = (5.53568306598135e-07 ~ 1.6607049197944e-06)[ic:1.1071366131962695e-06];
	param q_wq = (6.30366785348235e-06 ~ 1.8911003560447e-05)[ic:1.260733570696469e-05];
	param lpdbin = (5.87706470478477e-07 ~ 1.76311941143543e-06)[ic:1.1754129409569538e-06];
	param wpdbin = (4.90018510882979e-06 ~ 1.47005553264894e-05)[ic:9.80037021765958e-06];
	param lpdin = (4.44180050602804e-07 ~ 1.33254015180841e-06)[ic:8.883601012056073e-07];
	param wpdin = (2.54817987496434e-06 ~ 7.64453962489303e-06)[ic:5.0963597499286865e-06];
	param luumid = (4.84530757187268e-07 ~ 1.4535922715618e-06)[ic:9.69061514374535e-07];
	param wuumid = (3.53180538169263e-06 ~ 1.05954161450779e-05)[ic:7.063610763385259e-06];
	param lumid = (4.97814120781555e-07 ~ 1.49344236234467e-06)[ic:9.956282415631108e-07];
	param wumid = (6.18843335146195e-06 ~ 1.85653000543859e-05)[ic:1.2376866702923902e-05];
	param lp4 = (6.15296917657444e-07 ~ 1.84589075297233e-06)[ic:1.2305938353148878e-06];
	param wp4 = (6.70448167303965e-07 ~ 2.0113445019119e-06)[ic:1.340896334607931e-06];
	param ln4 = (6.56596138296711e-07 ~ 1.96978841489013e-06)[ic:1.313192276593423e-06];
	param wn4 = (1.63616707261929e-06 ~ 4.90850121785786e-06)[ic:3.2723341452385716e-06];
	param lnsupp = (9.74947340300978e-07 ~ 2.92484202090293e-06)[ic:1.9498946806019566e-06];
	param wnsupp = (1.9662800393469e-06 ~ 5.89884011804069e-06)[ic:3.932560078693794e-06];
	param lnsupp2 = (9.63499804044783e-07 ~ 2.89049941213435e-06)[ic:1.926999608089565e-06];
	param wnsupp2 = (1.51501787765634e-06 ~ 4.54505363296903e-06)[ic:3.0300357553126835e-06];
	param li10 = (2.81692665868799e-06 ~ 8.45077997606398e-06)[ic:5.633853317375985e-06];
	param wi10 = (1.49263475935951e-06 ~ 4.47790427807852e-06)[ic:2.9852695187190105e-06];
	param lb1 = (3.22907291632356e-06 ~ 9.68721874897068e-06)[ic:6.458145832647117e-06];
	param wb1 = (6.59169810729582e-06 ~ 1.97750943218875e-05)[ic:1.3183396214591634e-05];
	param lb2 = (3.51025381493195e-06 ~ 1.05307614447959e-05)[ic:7.020507629863902e-06];
	param wb2 = (1.2953023889796e-06 ~ 3.88590716693881e-06)[ic:2.5906047779592056e-06];
	param lb3 = (4.01047100523843e-07 ~ 1.20314130157153e-06)[ic:8.020942010476868e-07];
	param wb3 = (7.22990486162404e-07 ~ 2.16897145848721e-06)[ic:1.4459809723248086e-06];
	param lb4 = (1.35865956714392e-06 ~ 4.07597870143177e-06)[ic:2.7173191342878446e-06];
	param wb4 = (3.02441849312354e-06 ~ 9.07325547937061e-06)[ic:6.048836986247074e-06];
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
