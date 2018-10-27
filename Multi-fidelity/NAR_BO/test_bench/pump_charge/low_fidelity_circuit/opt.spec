folder::prj_dir is "/export/home/wllv/shzhang/Multi-fidelity/code/NAR_BO/test_bench/pump_charge/low_fidelity_circuit";
folder::abs_process_root is "/export/home/svnhome/process/smic.40/";
file::parameter is "charge_pump_param"      # here dcdc_param is the parameter file
{
    param_file_genre = "hspice";
	param q_llower = (1.5e-06 ~ 4.5e-06)[ic:3e-06];
	param q_wlower = (5.63468563621407e-07 ~ 1.69040569086422e-06)[ic:1.1269371272428141e-06];
	param q_lupper = (1e-06 ~ 3e-06)[ic:2e-06];
	param q_wupper = (5.7329438955464e-06 ~ 1.71988316866392e-05)[ic:1.1465887791092804e-05];
	param q_lc = (1.86020556237873e-06 ~ 5.58061668713618e-06)[ic:3.720411124757452e-06];
	param q_wc = (8.48446295980845e-06 ~ 2.54533888794254e-05)[ic:1.69689259196169e-05];
	param q_lref = (2e-06 ~ 6e-06)[ic:4e-06];
	param q_wref = (2.50000019310709e-06 ~ 7.50000057932128e-06)[ic:5.000000386214185e-06];
	param q_lq = (5.0000000000227e-07 ~ 1.50000000000681e-06)[ic:1.00000000000454e-06];
	param q_wq = (4.98652899412775e-06 ~ 1.49595869823833e-05)[ic:9.973057988255505e-06];
	param lpdbin = (9.75155332539831e-07 ~ 2.92546599761949e-06)[ic:1.950310665079663e-06];
	param wpdbin = (2.89167124981515e-06 ~ 8.67501374944544e-06)[ic:5.783342499630296e-06];
	param lpdin = (3.54257834410124e-07 ~ 1.06277350323037e-06)[ic:7.08515668820249e-07];
	param wpdin = (1.0000000142764e-06 ~ 3.00000004282921e-06)[ic:2.0000000285528097e-06];
	param luumid = (3e-07 ~ 9e-07)[ic:6e-07];
	param wuumid = (2.50000000073181e-06 ~ 7.50000000219542e-06)[ic:5.00000000146361e-06];
	param lumid = (2.75e-07 ~ 8.25e-07)[ic:5.500000000000001e-07];
	param wumid = (4.00000039019411e-06 ~ 1.20000011705823e-05)[ic:8.000000780388217e-06];
	param lp4 = (6.06657930819363e-07 ~ 1.81997379245809e-06)[ic:1.2133158616387267e-06];
	param wp4 = (5.0000576012015e-07 ~ 1.50001728036045e-06)[ic:1.000011520240301e-06];
	param ln4 = (1e-06 ~ 3e-06)[ic:2e-06];
	param wn4 = (5.00000045659747e-07 ~ 1.50000013697924e-06)[ic:1.0000000913194938e-06];
	param lnsupp = (1e-06 ~ 3e-06)[ic:2e-06];
	param wnsupp = (5.00004554647334e-07 ~ 1.500013663942e-06)[ic:1.000009109294668e-06];
	param lnsupp2 = (1.19999114926451e-06 ~ 3.59997344779353e-06)[ic:2.3999822985290227e-06];
	param wnsupp2 = (1.56665578615592e-06 ~ 4.69996735846777e-06)[ic:3.133311572311844e-06];
	param li10 = (1.00000000182402e-06 ~ 3.00000000547205e-06)[ic:2.0000000036480366e-06];
	param wi10 = (1.12290642127388e-06 ~ 3.36871926382164e-06)[ic:2.2458128425477584e-06];
	param lb1 = (3.89605563881489e-06 ~ 1.16881669164447e-05)[ic:7.792111277629786e-06];
	param wb1 = (1.25e-05 ~ 3.75e-05)[ic:2.5e-05];
	param lb2 = (1.00000105617617e-06 ~ 3.00000316852852e-06)[ic:2.0000021123523466e-06];
	param wb2 = (5.00054809518435e-07 ~ 1.5001644285553e-06)[ic:1.0001096190368699e-06];
	param lb3 = (5.91556496758928e-07 ~ 1.77466949027678e-06)[ic:1.1831129935178558e-06];
	param wb3 = (5.03319187407028e-07 ~ 1.50995756222108e-06)[ic:1.0066383748140562e-06];
	param lb4 = (1.5e-06 ~ 4.5e-06)[ic:3e-06];
	param wb4 = (2.00000000116784e-06 ~ 6.00000000350353e-06)[ic:4.0000000023356855e-06];
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
