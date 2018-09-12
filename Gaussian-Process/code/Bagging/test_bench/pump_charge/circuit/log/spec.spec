folder::prj_dir is "/export/home/wllv/shzhang/Bagging/test_bench/test3/circuit";
folder::abs_process_root is "/export/home/svnhome/process/smic.40/";
file::parameter is "charge_pump_param"      # here dcdc_param is the parameter file
{
    param_file_genre = "hspice";
	param q_llower = (3.78390770113944e-06 ~ 1.13517231034183e-05)[ic:7.567815402278882e-06];
	param q_wlower = (1.13666682274418e-06 ~ 3.41000046823255e-06)[ic:2.273333645488365e-06];
	param q_lupper = (7.25988911884738e-07 ~ 2.17796673565422e-06)[ic:1.4519778237694767e-06];
	param q_wupper = (4.08474725301432e-06 ~ 1.2254241759043e-05)[ic:8.169494506028637e-06];
	param q_lc = (9.49285130145284e-07 ~ 2.84785539043585e-06)[ic:1.8985702602905688e-06];
	param q_wc = (3.98560962235128e-06 ~ 1.19568288670538e-05)[ic:7.971219244702563e-06];
	param q_lref = (7.32813588299578e-07 ~ 2.19844076489873e-06)[ic:1.465627176599156e-06];
	param q_wref = (8.90046066499901e-06 ~ 2.6701381994997e-05)[ic:1.7800921329998014e-05];
	param q_lq = (1.75394884774729e-06 ~ 5.26184654324186e-06)[ic:3.5078976954945712e-06];
	param q_wq = (5.51935286167203e-06 ~ 1.65580585850161e-05)[ic:1.103870572334406e-05];
	param lpdbin = (8.29392581752498e-07 ~ 2.48817774525749e-06)[ic:1.6587851635049964e-06];
	param wpdbin = (3.80048618888646e-06 ~ 1.14014585666594e-05)[ic:7.6009723777729275e-06];
	param lpdin = (2.76383297559647e-07 ~ 8.29149892678942e-07)[ic:5.527665951192945e-07];
	param wpdin = (2.50478700119477e-06 ~ 7.51436100358431e-06)[ic:5.009574002389539e-06];
	param luumid = (5.8921427011797e-07 ~ 1.76764281035391e-06)[ic:1.17842854023594e-06];
	param wuumid = (9.4682168551623e-06 ~ 2.84046505654869e-05)[ic:1.8936433710324602e-05];
	param lumid = (6.98143618340823e-07 ~ 2.09443085502247e-06)[ic:1.396287236681647e-06];
	param wumid = (9.99849566042127e-06 ~ 2.99954869812638e-05)[ic:1.999699132084255e-05];
	param lp4 = (8.1063837143825e-07 ~ 2.43191511431475e-06)[ic:1.6212767428764993e-06];
	param wp4 = (5.99010808786268e-07 ~ 1.7970324263588e-06)[ic:1.1980216175725356e-06];
	param ln4 = (6.17862392611132e-07 ~ 1.8535871778334e-06)[ic:1.2357247852222643e-06];
	param wn4 = (8.31902940768303e-07 ~ 2.49570882230491e-06)[ic:1.6638058815366057e-06];
	param lnsupp = (7.44792704874894e-07 ~ 2.23437811462468e-06)[ic:1.489585409749789e-06];
	param wnsupp = (8.55894394746238e-07 ~ 2.56768318423871e-06)[ic:1.7117887894924753e-06];
	param lnsupp2 = (7.59953893839223e-07 ~ 2.27986168151767e-06)[ic:1.5199077876784455e-06];
	param wnsupp2 = (1.44484750827911e-06 ~ 4.33454252483733e-06)[ic:2.8896950165582203e-06];
	param li10 = (1.06407594733884e-06 ~ 3.19222784201651e-06)[ic:2.1281518946776704e-06];
	param wi10 = (8.80478039520026e-07 ~ 2.64143411856008e-06)[ic:1.7609560790400517e-06];
	param lb1 = (3.72319930742301e-06 ~ 1.1169597922269e-05)[ic:7.446398614846024e-06];
	param wb1 = (7.07012141437762e-06 ~ 2.12103642431329e-05)[ic:1.4140242828755246e-05];
	param lb2 = (3.01042075816573e-06 ~ 9.03126227449719e-06)[ic:6.020841516331461e-06];
	param wb2 = (6.92112893419259e-07 ~ 2.07633868025778e-06)[ic:1.3842257868385175e-06];
	param lb3 = (9.67602637403631e-07 ~ 2.90280791221089e-06)[ic:1.935205274807262e-06];
	param wb3 = (7.27403606203628e-07 ~ 2.18221081861088e-06)[ic:1.4548072124072566e-06];
	param lb4 = (8.98388495322468e-07 ~ 2.6951654859674e-06)[ic:1.796776990644936e-06];
	param wb4 = (4.35744622781524e-06 ~ 1.30723386834457e-05)[ic:8.714892455630488e-06];
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
