OPENQASM 2.0;
include "hqslib1.inc";

qreg q[40];
creg c[40];
U1q(1.78781173028699*pi,1.8151981500222591*pi) q[0];
U1q(1.53304530656073*pi,0.5835751541780492*pi) q[1];
U1q(1.40130645169329*pi,0.13591290243028606*pi) q[2];
U1q(0.399133481074199*pi,0.698482298724291*pi) q[3];
U1q(0.86887557545426*pi,1.43747536618865*pi) q[4];
U1q(1.63287286100338*pi,1.3908275551357248*pi) q[5];
U1q(0.506091298252687*pi,1.24756948090409*pi) q[6];
U1q(1.49435374470117*pi,1.0394471363361784*pi) q[7];
U1q(0.609895534740445*pi,1.35787354523652*pi) q[8];
U1q(0.45434661734571*pi,0.384699571435296*pi) q[9];
U1q(1.70238532467067*pi,1.3486790334939087*pi) q[10];
U1q(1.63802392449093*pi,0.301362066528845*pi) q[11];
U1q(1.21941009518572*pi,1.4168001311670966*pi) q[12];
U1q(1.65597938651772*pi,1.0815187975716927*pi) q[13];
U1q(0.72183909648314*pi,1.3933583379079701*pi) q[14];
U1q(0.373673196568628*pi,0.82273414162255*pi) q[15];
U1q(1.48480654950236*pi,0.7816822101227517*pi) q[16];
U1q(0.416347298202812*pi,1.148869028279468*pi) q[17];
U1q(1.71878766011372*pi,0.05156444672073584*pi) q[18];
U1q(1.50462316662985*pi,1.092294459495799*pi) q[19];
U1q(0.519077040986467*pi,1.30627236557406*pi) q[20];
U1q(1.40326222507496*pi,1.760649022272377*pi) q[21];
U1q(3.739512995412619*pi,0.9190327408785167*pi) q[22];
U1q(0.722914148795713*pi,1.083093998238696*pi) q[23];
U1q(0.625445600694927*pi,1.031236005878841*pi) q[24];
U1q(0.559949584603556*pi,0.489575942038296*pi) q[25];
U1q(1.83531645036611*pi,1.8276594404136959*pi) q[26];
U1q(0.68326633912479*pi,0.586912654349226*pi) q[27];
U1q(0.709275492735376*pi,1.9260902391468713*pi) q[28];
U1q(0.658647414317733*pi,1.5551293115797*pi) q[29];
U1q(0.86220771571037*pi,0.557200131623699*pi) q[30];
U1q(1.28108905046756*pi,0.962274489901679*pi) q[31];
U1q(1.74066481961448*pi,0.11573401951854606*pi) q[32];
U1q(1.15204225148832*pi,1.8215970333159313*pi) q[33];
U1q(1.76615010755266*pi,0.4000822897268866*pi) q[34];
U1q(0.671494676920924*pi,0.828190143072867*pi) q[35];
U1q(0.100074203151483*pi,0.9392473765909499*pi) q[36];
U1q(0.111370497341832*pi,1.4782212333865061*pi) q[37];
U1q(1.60151243499047*pi,0.009408218719879435*pi) q[38];
U1q(0.350328292979629*pi,1.08583471298356*pi) q[39];
RZZ(0.5*pi) q[0],q[10];
RZZ(0.5*pi) q[13],q[1];
RZZ(0.5*pi) q[9],q[2];
RZZ(0.5*pi) q[5],q[3];
RZZ(0.5*pi) q[21],q[4];
RZZ(0.5*pi) q[6],q[35];
RZZ(0.5*pi) q[22],q[7];
RZZ(0.5*pi) q[8],q[29];
RZZ(0.5*pi) q[11],q[36];
RZZ(0.5*pi) q[12],q[20];
RZZ(0.5*pi) q[34],q[14];
RZZ(0.5*pi) q[24],q[15];
RZZ(0.5*pi) q[16],q[17];
RZZ(0.5*pi) q[18],q[32];
RZZ(0.5*pi) q[25],q[19];
RZZ(0.5*pi) q[30],q[23];
RZZ(0.5*pi) q[26],q[27];
RZZ(0.5*pi) q[37],q[28];
RZZ(0.5*pi) q[31],q[39];
RZZ(0.5*pi) q[33],q[38];
U1q(0.341278463017108*pi,0.46182223019804924*pi) q[0];
U1q(0.46251468169132*pi,1.3596131429807392*pi) q[1];
U1q(0.68824103118901*pi,0.37526973467578606*pi) q[2];
U1q(0.410904766173529*pi,1.06814020347101*pi) q[3];
U1q(0.326034934899161*pi,0.64868651472168*pi) q[4];
U1q(0.271866276536637*pi,0.646202773514895*pi) q[5];
U1q(0.327045107885552*pi,0.045326586111812*pi) q[6];
U1q(0.540001598597716*pi,0.023860302664268307*pi) q[7];
U1q(0.730845852290676*pi,0.7486132714711302*pi) q[8];
U1q(0.748595005053257*pi,0.03367559452818991*pi) q[9];
U1q(0.341967218466699*pi,1.854971061651419*pi) q[10];
U1q(0.411823553841128*pi,0.7551214249560549*pi) q[11];
U1q(0.440162488512591*pi,0.5123940300005065*pi) q[12];
U1q(0.785261930903087*pi,0.6935904268001725*pi) q[13];
U1q(0.360091607152854*pi,0.7760707636319499*pi) q[14];
U1q(0.198751267065631*pi,0.8689645992591402*pi) q[15];
U1q(0.67413425855977*pi,1.4675728666267416*pi) q[16];
U1q(0.821031704575141*pi,1.3417964444747899*pi) q[17];
U1q(0.674652925137119*pi,0.07908556711593606*pi) q[18];
U1q(0.525856893578696*pi,0.355784267981909*pi) q[19];
U1q(0.518532061895211*pi,0.4125884734283001*pi) q[20];
U1q(0.657204118447261*pi,0.9947516336300368*pi) q[21];
U1q(0.535192369133439*pi,1.1664150037524066*pi) q[22];
U1q(0.595610893452803*pi,1.6659602121825596*pi) q[23];
U1q(0.522559661748754*pi,0.32849429792053986*pi) q[24];
U1q(0.662465182203159*pi,0.47559440706305*pi) q[25];
U1q(0.41384833894162*pi,0.4778982188338059*pi) q[26];
U1q(0.798995759728245*pi,0.35666316591247993*pi) q[27];
U1q(0.417162120619061*pi,1.2650035901520296*pi) q[28];
U1q(0.715349302987809*pi,0.0623386619748256*pi) q[29];
U1q(0.673127046702529*pi,1.41600312707576*pi) q[30];
U1q(0.32858662098149*pi,0.630341191014129*pi) q[31];
U1q(0.0873206621756453*pi,0.010426232128935897*pi) q[32];
U1q(0.343488991909178*pi,1.815394244421661*pi) q[33];
U1q(0.808075073571473*pi,1.9358348708369766*pi) q[34];
U1q(0.590288633953656*pi,1.636100665418037*pi) q[35];
U1q(0.899119144671088*pi,1.68494614565331*pi) q[36];
U1q(0.403432019055004*pi,1.1502697040575698*pi) q[37];
U1q(0.537150652442785*pi,0.6487068965712195*pi) q[38];
U1q(0.133400995355367*pi,1.8161580902289*pi) q[39];
RZZ(0.5*pi) q[1],q[0];
RZZ(0.5*pi) q[33],q[2];
RZZ(0.5*pi) q[3],q[35];
RZZ(0.5*pi) q[31],q[4];
RZZ(0.5*pi) q[5],q[23];
RZZ(0.5*pi) q[6],q[24];
RZZ(0.5*pi) q[32],q[7];
RZZ(0.5*pi) q[8],q[10];
RZZ(0.5*pi) q[9],q[14];
RZZ(0.5*pi) q[20],q[11];
RZZ(0.5*pi) q[12],q[18];
RZZ(0.5*pi) q[13],q[30];
RZZ(0.5*pi) q[29],q[15];
RZZ(0.5*pi) q[16],q[22];
RZZ(0.5*pi) q[38],q[17];
RZZ(0.5*pi) q[19],q[36];
RZZ(0.5*pi) q[28],q[21];
RZZ(0.5*pi) q[25],q[34];
RZZ(0.5*pi) q[37],q[26];
RZZ(0.5*pi) q[39],q[27];
U1q(0.357504609416186*pi,1.3425278562138994*pi) q[0];
U1q(0.876475945451664*pi,0.3051063899720394*pi) q[1];
U1q(0.879362256008925*pi,0.5972235200665859*pi) q[2];
U1q(0.468806448258667*pi,0.5736002606084902*pi) q[3];
U1q(0.916634029445668*pi,0.15987035010446005*pi) q[4];
U1q(0.320859834936461*pi,0.5401204253472747*pi) q[5];
U1q(0.457349392143543*pi,0.5033916184648799*pi) q[6];
U1q(0.39505659156553*pi,0.22965797687560752*pi) q[7];
U1q(0.224919765293276*pi,0.27132369889771013*pi) q[8];
U1q(0.398739129563162*pi,1.7158511416990399*pi) q[9];
U1q(0.405391852294818*pi,1.4515306644214183*pi) q[10];
U1q(0.591285095833847*pi,0.4278807100134152*pi) q[11];
U1q(0.950186161423709*pi,1.9374510499305968*pi) q[12];
U1q(0.437833205592807*pi,1.4185307238724825*pi) q[13];
U1q(0.13796101334021*pi,1.7634167657936297*pi) q[14];
U1q(0.326897162320509*pi,1.5873741230873204*pi) q[15];
U1q(0.747639383236022*pi,1.6970116873535117*pi) q[16];
U1q(0.59286171298959*pi,0.11459967122365011*pi) q[17];
U1q(0.321704004673709*pi,1.0815764367974863*pi) q[18];
U1q(0.180875628319732*pi,0.47477312716851827*pi) q[19];
U1q(0.397277722284645*pi,0.14991869397350044*pi) q[20];
U1q(0.461620165872279*pi,0.3979843856906964*pi) q[21];
U1q(0.608208310671334*pi,1.687507968712707*pi) q[22];
U1q(0.233492702428056*pi,1.1009169874121199*pi) q[23];
U1q(0.629368919921577*pi,0.7481599761272202*pi) q[24];
U1q(0.319322374119799*pi,0.6157822061199201*pi) q[25];
U1q(0.329008099167084*pi,0.7488986443218657*pi) q[26];
U1q(0.531018173693993*pi,0.9503001196230496*pi) q[27];
U1q(0.255622982696878*pi,0.4109624582153497*pi) q[28];
U1q(0.879688814512105*pi,1.5239032780209798*pi) q[29];
U1q(0.901042609891977*pi,0.8656705584475004*pi) q[30];
U1q(0.115482479098627*pi,1.8309982838658287*pi) q[31];
U1q(0.422840657526715*pi,0.5938873559109554*pi) q[32];
U1q(0.514697122472278*pi,0.07979261389977133*pi) q[33];
U1q(0.468473609961474*pi,0.6735029357273268*pi) q[34];
U1q(0.871656303547799*pi,0.4367526726961599*pi) q[35];
U1q(0.379043257363075*pi,0.7497199988961496*pi) q[36];
U1q(0.667907260450572*pi,0.5078061187129301*pi) q[37];
U1q(0.647012991303246*pi,1.0366506795566197*pi) q[38];
U1q(0.483275106988619*pi,1.5248810427993202*pi) q[39];
RZZ(0.5*pi) q[16],q[0];
RZZ(0.5*pi) q[1],q[8];
RZZ(0.5*pi) q[17],q[2];
RZZ(0.5*pi) q[3],q[14];
RZZ(0.5*pi) q[26],q[4];
RZZ(0.5*pi) q[5],q[37];
RZZ(0.5*pi) q[6],q[7];
RZZ(0.5*pi) q[31],q[9];
RZZ(0.5*pi) q[10],q[29];
RZZ(0.5*pi) q[38],q[11];
RZZ(0.5*pi) q[12],q[39];
RZZ(0.5*pi) q[13],q[28];
RZZ(0.5*pi) q[15],q[27];
RZZ(0.5*pi) q[18],q[21];
RZZ(0.5*pi) q[23],q[19];
RZZ(0.5*pi) q[20],q[30];
RZZ(0.5*pi) q[22],q[25];
RZZ(0.5*pi) q[34],q[24];
RZZ(0.5*pi) q[36],q[32];
RZZ(0.5*pi) q[33],q[35];
U1q(0.474065294820011*pi,0.727158224216959*pi) q[0];
U1q(0.425622186914615*pi,0.4295523138293884*pi) q[1];
U1q(0.373458529744435*pi,1.138302445562136*pi) q[2];
U1q(0.354172700269049*pi,1.2938772488052797*pi) q[3];
U1q(0.516728424219876*pi,1.9005283740267096*pi) q[4];
U1q(0.507639839635108*pi,0.7593205220103645*pi) q[5];
U1q(0.783741250318545*pi,1.5211801417051696*pi) q[6];
U1q(0.0927530879058959*pi,0.1608279942398081*pi) q[7];
U1q(0.511065399368478*pi,1.7777588574308005*pi) q[8];
U1q(0.628724292725209*pi,0.7736511340007004*pi) q[9];
U1q(0.865154464402489*pi,0.5786034844145389*pi) q[10];
U1q(0.632945479435573*pi,0.8081938047073844*pi) q[11];
U1q(0.562145917171512*pi,1.6110362290468867*pi) q[12];
U1q(0.880002833563369*pi,1.0138032147241027*pi) q[13];
U1q(0.414834649506175*pi,1.1061471316116602*pi) q[14];
U1q(0.183458847059727*pi,0.13861426169196012*pi) q[15];
U1q(0.849610805851052*pi,0.629144003216342*pi) q[16];
U1q(0.376038862179641*pi,0.8152561169953598*pi) q[17];
U1q(0.613348148114529*pi,0.18293546806772554*pi) q[18];
U1q(0.513369835892216*pi,0.22023538394715914*pi) q[19];
U1q(0.671207587011816*pi,0.13143322347591013*pi) q[20];
U1q(0.536001000064195*pi,1.1750997133837462*pi) q[21];
U1q(0.785994215479756*pi,1.9472583789986766*pi) q[22];
U1q(0.173598800000181*pi,1.5139416676615003*pi) q[23];
U1q(0.460824447173799*pi,0.29827204608954005*pi) q[24];
U1q(0.353513132733405*pi,1.3224117153188697*pi) q[25];
U1q(0.330520414879871*pi,1.2092292444061457*pi) q[26];
U1q(0.871728286927765*pi,0.5320941864646098*pi) q[27];
U1q(0.49481507863108*pi,0.8669160281624393*pi) q[28];
U1q(0.666649590714049*pi,0.2283358843365102*pi) q[29];
U1q(0.549476933803609*pi,1.4790664999629701*pi) q[30];
U1q(0.640755578337248*pi,0.34295588897279927*pi) q[31];
U1q(0.663524264654036*pi,1.510735017013106*pi) q[32];
U1q(0.295042860557264*pi,1.2917840309455126*pi) q[33];
U1q(0.389284874625567*pi,1.2788040598337966*pi) q[34];
U1q(0.762902851057922*pi,0.6344500044606596*pi) q[35];
U1q(0.792841723545909*pi,1.6617483583706996*pi) q[36];
U1q(0.579207130427438*pi,0.44899244117397963*pi) q[37];
U1q(0.283125091779906*pi,0.5749309152883288*pi) q[38];
U1q(0.360709189653156*pi,0.09708005640666961*pi) q[39];
RZZ(0.5*pi) q[0],q[27];
RZZ(0.5*pi) q[1],q[39];
RZZ(0.5*pi) q[23],q[2];
RZZ(0.5*pi) q[3],q[30];
RZZ(0.5*pi) q[18],q[4];
RZZ(0.5*pi) q[5],q[6];
RZZ(0.5*pi) q[11],q[7];
RZZ(0.5*pi) q[37],q[8];
RZZ(0.5*pi) q[33],q[9];
RZZ(0.5*pi) q[12],q[10];
RZZ(0.5*pi) q[13],q[26];
RZZ(0.5*pi) q[20],q[14];
RZZ(0.5*pi) q[34],q[15];
RZZ(0.5*pi) q[16],q[31];
RZZ(0.5*pi) q[17],q[32];
RZZ(0.5*pi) q[38],q[19];
RZZ(0.5*pi) q[21],q[35];
RZZ(0.5*pi) q[22],q[36];
RZZ(0.5*pi) q[25],q[24];
RZZ(0.5*pi) q[28],q[29];
U1q(0.787539638633116*pi,1.495927421795539*pi) q[0];
U1q(0.440003291921581*pi,0.35083521185847033*pi) q[1];
U1q(0.746975692618682*pi,1.7420207995598869*pi) q[2];
U1q(0.479745154337947*pi,0.05287628294538038*pi) q[3];
U1q(0.567206060254481*pi,0.9391763948725202*pi) q[4];
U1q(0.401087366766405*pi,0.9836869161927257*pi) q[5];
U1q(0.687046219402872*pi,1.8192634571889101*pi) q[6];
U1q(0.612312909109349*pi,0.7833065895800679*pi) q[7];
U1q(0.929766511650504*pi,0.6893223156295001*pi) q[8];
U1q(0.516650180001799*pi,1.7310918526643206*pi) q[9];
U1q(0.385197642284151*pi,0.22409553274903793*pi) q[10];
U1q(0.39677140495906*pi,1.602917761932865*pi) q[11];
U1q(0.465400135342593*pi,0.9814187998183979*pi) q[12];
U1q(0.372905563721825*pi,1.272881047779693*pi) q[13];
U1q(0.131784555135519*pi,1.4730078960356003*pi) q[14];
U1q(0.460593151727442*pi,0.7047150200487504*pi) q[15];
U1q(0.739100080870951*pi,0.22028134313804149*pi) q[16];
U1q(0.472931455209003*pi,1.1331316338372108*pi) q[17];
U1q(0.872766465897933*pi,0.10209955905693668*pi) q[18];
U1q(0.951335957025577*pi,1.9633388004637293*pi) q[19];
U1q(0.756681739836407*pi,1.4749268201450008*pi) q[20];
U1q(0.546916801095603*pi,0.5542827922973874*pi) q[21];
U1q(0.827855086088942*pi,0.3665221215836363*pi) q[22];
U1q(0.225677145116131*pi,0.7408652818949992*pi) q[23];
U1q(0.419997598533064*pi,1.1027161311449305*pi) q[24];
U1q(0.356012008461168*pi,1.2750689185642496*pi) q[25];
U1q(0.626949423059324*pi,0.989009436605766*pi) q[26];
U1q(0.309306093748583*pi,0.9056996181355998*pi) q[27];
U1q(0.928823767711817*pi,1.6993393947052997*pi) q[28];
U1q(0.542847295180658*pi,1.69938147059307*pi) q[29];
U1q(0.557193768177356*pi,1.77039036512336*pi) q[30];
U1q(0.23093614508654*pi,1.600782639446809*pi) q[31];
U1q(0.860314302779595*pi,1.152866372077706*pi) q[32];
U1q(0.482728266387227*pi,1.7110775161161325*pi) q[33];
U1q(0.667285751859603*pi,0.2920312717883258*pi) q[34];
U1q(0.355604183976618*pi,0.8122488148457698*pi) q[35];
U1q(0.561312629263255*pi,1.5826559499655009*pi) q[36];
U1q(0.376317411143557*pi,1.0818595453552797*pi) q[37];
U1q(0.230477290484073*pi,0.03636879270710924*pi) q[38];
U1q(0.757497539507953*pi,1.4447498879570393*pi) q[39];
RZZ(0.5*pi) q[0],q[35];
RZZ(0.5*pi) q[26],q[1];
RZZ(0.5*pi) q[38],q[2];
RZZ(0.5*pi) q[3],q[8];
RZZ(0.5*pi) q[25],q[4];
RZZ(0.5*pi) q[5],q[27];
RZZ(0.5*pi) q[6],q[36];
RZZ(0.5*pi) q[24],q[7];
RZZ(0.5*pi) q[9],q[32];
RZZ(0.5*pi) q[37],q[10];
RZZ(0.5*pi) q[33],q[11];
RZZ(0.5*pi) q[12],q[17];
RZZ(0.5*pi) q[13],q[31];
RZZ(0.5*pi) q[28],q[14];
RZZ(0.5*pi) q[20],q[15];
RZZ(0.5*pi) q[16],q[21];
RZZ(0.5*pi) q[18],q[30];
RZZ(0.5*pi) q[34],q[19];
RZZ(0.5*pi) q[22],q[39];
RZZ(0.5*pi) q[29],q[23];
U1q(0.106463929471562*pi,0.7472082721978577*pi) q[0];
U1q(0.410092974007126*pi,0.7607669726829194*pi) q[1];
U1q(0.635411710828253*pi,0.8048693269359859*pi) q[2];
U1q(0.450967800597411*pi,1.0852686829059*pi) q[3];
U1q(0.213633655308944*pi,1.0508189896637*pi) q[4];
U1q(0.656560757420369*pi,0.14822535625322608*pi) q[5];
U1q(0.248939918556761*pi,1.8816986115853709*pi) q[6];
U1q(0.414227104603034*pi,1.2125195521526493*pi) q[7];
U1q(0.559094704504791*pi,0.7951054847424004*pi) q[8];
U1q(0.109424544268496*pi,1.5628279678366006*pi) q[9];
U1q(0.417778650304923*pi,1.261997901223209*pi) q[10];
U1q(0.237668705994018*pi,0.7929423711065162*pi) q[11];
U1q(0.796877946283995*pi,1.7358407632363964*pi) q[12];
U1q(0.540680318109576*pi,1.4481415021723922*pi) q[13];
U1q(0.717404432598523*pi,0.13999999037750044*pi) q[14];
U1q(0.133587831994096*pi,0.3812327282442993*pi) q[15];
U1q(0.48396020827398*pi,1.1375480221545509*pi) q[16];
U1q(0.718678209723504*pi,0.44143519241744045*pi) q[17];
U1q(0.106021857307457*pi,1.150127222147736*pi) q[18];
U1q(0.680088475588226*pi,0.2128787059231989*pi) q[19];
U1q(0.381133829402488*pi,1.5750376402508*pi) q[20];
U1q(0.615462157914779*pi,0.35413268450052726*pi) q[21];
U1q(0.674032803094244*pi,0.3837112073591662*pi) q[22];
U1q(0.248869437722238*pi,1.6519593850259007*pi) q[23];
U1q(0.0282789392787125*pi,1.4823583292756002*pi) q[24];
U1q(0.459740803953731*pi,1.5288079906029992*pi) q[25];
U1q(0.385637043562059*pi,0.2603429482221955*pi) q[26];
U1q(0.447088388991269*pi,0.5824723700168004*pi) q[27];
U1q(0.362763160896594*pi,1.6070735920922008*pi) q[28];
U1q(0.130593706084358*pi,0.07549194251244007*pi) q[29];
U1q(0.342929546029622*pi,0.3897255824234005*pi) q[30];
U1q(0.258508050663265*pi,0.8351369330563791*pi) q[31];
U1q(0.962056202824317*pi,1.531641296698627*pi) q[32];
U1q(0.741549576580352*pi,1.002504120874832*pi) q[33];
U1q(0.402608310910171*pi,1.638565630218988*pi) q[34];
U1q(0.75807348407041*pi,0.7700189380325*pi) q[35];
U1q(0.199216371685909*pi,1.9795863587686*pi) q[36];
U1q(0.442477205199471*pi,0.5579344243956399*pi) q[37];
U1q(0.493016960037187*pi,1.5218056586321804*pi) q[38];
U1q(0.105051613698539*pi,0.4744285197456204*pi) q[39];
RZZ(0.5*pi) q[31],q[0];
RZZ(0.5*pi) q[33],q[1];
RZZ(0.5*pi) q[35],q[2];
RZZ(0.5*pi) q[3],q[6];
RZZ(0.5*pi) q[4],q[24];
RZZ(0.5*pi) q[5],q[13];
RZZ(0.5*pi) q[39],q[7];
RZZ(0.5*pi) q[26],q[8];
RZZ(0.5*pi) q[9],q[28];
RZZ(0.5*pi) q[10],q[20];
RZZ(0.5*pi) q[37],q[11];
RZZ(0.5*pi) q[12],q[14];
RZZ(0.5*pi) q[19],q[15];
RZZ(0.5*pi) q[16],q[25];
RZZ(0.5*pi) q[22],q[17];
RZZ(0.5*pi) q[18],q[27];
RZZ(0.5*pi) q[29],q[21];
RZZ(0.5*pi) q[23],q[32];
RZZ(0.5*pi) q[34],q[30];
RZZ(0.5*pi) q[38],q[36];
U1q(0.683319360268926*pi,1.4059314932028588*pi) q[0];
U1q(0.748913653962935*pi,1.9551039131909498*pi) q[1];
U1q(0.513950773521337*pi,0.09637419261878577*pi) q[2];
U1q(0.506174674655129*pi,1.031701318964*pi) q[3];
U1q(0.640037388662732*pi,1.4927555452131998*pi) q[4];
U1q(0.352373808742532*pi,1.320427913580625*pi) q[5];
U1q(0.585825119236574*pi,0.8861394481266007*pi) q[6];
U1q(0.183201125766507*pi,1.9744692436821794*pi) q[7];
U1q(0.360355017038078*pi,0.6635310073548997*pi) q[8];
U1q(0.633905337318893*pi,0.8875706761188003*pi) q[9];
U1q(0.758510890233495*pi,1.8869850744269083*pi) q[10];
U1q(0.531356917643515*pi,1.8657311600775461*pi) q[11];
U1q(0.68221399167027*pi,1.1344501231724955*pi) q[12];
U1q(0.798792517143955*pi,1.3863936882976926*pi) q[13];
U1q(0.357879585782896*pi,1.5305535281631997*pi) q[14];
U1q(0.715175200148395*pi,1.7424348260904008*pi) q[15];
U1q(0.539935865096316*pi,0.8711008076138516*pi) q[16];
U1q(0.494257563461887*pi,1.5370718161109007*pi) q[17];
U1q(0.433867823950405*pi,1.648840584575936*pi) q[18];
U1q(0.565851139014881*pi,0.3589442940477987*pi) q[19];
U1q(0.114959150052452*pi,0.03939984578470046*pi) q[20];
U1q(0.346361746647283*pi,1.6637048433289774*pi) q[21];
U1q(0.557677085837104*pi,0.8360672435690573*pi) q[22];
U1q(0.498914172075262*pi,0.9471966835326988*pi) q[23];
U1q(0.682635188637083*pi,0.6380109163306003*pi) q[24];
U1q(0.497277701122197*pi,1.9668585152302995*pi) q[25];
U1q(0.13486900889369*pi,0.5893875797318948*pi) q[26];
U1q(0.80313345186984*pi,1.7527310214667011*pi) q[27];
U1q(0.840111340275334*pi,0.6207560826432008*pi) q[28];
U1q(0.224291737687834*pi,1.6919528698436004*pi) q[29];
U1q(0.358069829067064*pi,1.8230583750106995*pi) q[30];
U1q(0.389635412667422*pi,0.28724069607557823*pi) q[31];
U1q(0.618987738202376*pi,1.3500815711365473*pi) q[32];
U1q(0.613649980407373*pi,1.0233426549466316*pi) q[33];
U1q(0.170138560388493*pi,0.6958708419930879*pi) q[34];
U1q(0.245257423116446*pi,1.2216291588703996*pi) q[35];
U1q(0.652797122212475*pi,0.9702499114772003*pi) q[36];
U1q(0.0809273187705369*pi,1.1316120064107995*pi) q[37];
U1q(0.513299859636652*pi,1.0378097511866802*pi) q[38];
U1q(0.823922560340513*pi,1.6630233078402998*pi) q[39];
rz(2.019249317672841*pi) q[0];
rz(3.6848493076984496*pi) q[1];
rz(1.5861855575465142*pi) q[2];
rz(1.2825492039692996*pi) q[3];
rz(3.816333021819201*pi) q[4];
rz(2.634749375149175*pi) q[5];
rz(3.9710504364228*pi) q[6];
rz(2.1270469600449218*pi) q[7];
rz(0.2124067452484013*pi) q[8];
rz(1.0678877524638999*pi) q[9];
rz(1.911720471329991*pi) q[10];
rz(2.047968660446754*pi) q[11];
rz(1.6602662842628035*pi) q[12];
rz(2.7736539033731074*pi) q[13];
rz(0.3114691674196983*pi) q[14];
rz(3.800769050670599*pi) q[15];
rz(0.14766633082204805*pi) q[16];
rz(2.8996921115569005*pi) q[17];
rz(1.9570891443712632*pi) q[18];
rz(0.8506966147522022*pi) q[19];
rz(1.7113597602447008*pi) q[20];
rz(0.42695141700382244*pi) q[21];
rz(2.8835019498073233*pi) q[22];
rz(3.510004755176901*pi) q[23];
rz(2.0124938045097007*pi) q[24];
rz(2.3102978278781006*pi) q[25];
rz(2.206139631851304*pi) q[26];
rz(0.8839069856580011*pi) q[27];
rz(1.2846371236487997*pi) q[28];
rz(3.2763162136724704*pi) q[29];
rz(3.0039783705203007*pi) q[30];
rz(1.5987643574091202*pi) q[31];
rz(2.202619473093254*pi) q[32];
rz(2.843799011917069*pi) q[33];
rz(1.2922510846734134*pi) q[34];
rz(1.7097953676339*pi) q[35];
rz(1.9618510066161008*pi) q[36];
rz(3.6344246561451996*pi) q[37];
rz(3.4968470891297194*pi) q[38];
rz(0.36802425155759977*pi) q[39];
U1q(0.683319360268926*pi,0.425180810875635*pi) q[0];
U1q(0.748913653962935*pi,0.639953220889455*pi) q[1];
U1q(1.51395077352134*pi,0.682559750165307*pi) q[2];
U1q(0.506174674655129*pi,1.314250522933315*pi) q[3];
U1q(0.640037388662732*pi,0.309088567032396*pi) q[4];
U1q(0.352373808742532*pi,0.95517728872988*pi) q[5];
U1q(1.58582511923657*pi,1.857189884549366*pi) q[6];
U1q(0.183201125766507*pi,1.101516203727166*pi) q[7];
U1q(1.36035501703808*pi,1.875937752603292*pi) q[8];
U1q(1.63390533731889*pi,0.955458428582698*pi) q[9];
U1q(1.7585108902335*pi,0.798705545756872*pi) q[10];
U1q(1.53135691764352*pi,0.913699820524376*pi) q[11];
U1q(0.68221399167027*pi,1.794716407435325*pi) q[12];
U1q(1.79879251714396*pi,1.16004759167076*pi) q[13];
U1q(0.357879585782896*pi,0.842022695582878*pi) q[14];
U1q(0.715175200148395*pi,0.543203876760976*pi) q[15];
U1q(0.539935865096316*pi,0.0187671384358512*pi) q[16];
U1q(1.49425756346189*pi,1.43676392766782*pi) q[17];
U1q(1.43386782395041*pi,0.605929728947163*pi) q[18];
U1q(3.565851139014881*pi,0.2096409088*pi) q[19];
U1q(0.114959150052452*pi,0.75075960602938*pi) q[20];
U1q(1.34636174664728*pi,1.090656260332782*pi) q[21];
U1q(0.557677085837104*pi,0.719569193376376*pi) q[22];
U1q(0.498914172075262*pi,1.457201438709655*pi) q[23];
U1q(1.68263518863708*pi,1.65050472084028*pi) q[24];
U1q(1.4972777011222*pi,1.2771563431084*pi) q[25];
U1q(0.13486900889369*pi,1.795527211583198*pi) q[26];
U1q(1.80313345186984*pi,1.636638007124758*pi) q[27];
U1q(1.84011134027533*pi,0.905393206292068*pi) q[28];
U1q(1.22429173768783*pi,1.9682690835160677*pi) q[29];
U1q(3.358069829067064*pi,1.827036745530963*pi) q[30];
U1q(1.38963541266742*pi,0.8860050534846899*pi) q[31];
U1q(1.61898773820238*pi,0.552701044229805*pi) q[32];
U1q(1.61364998040737*pi,0.86714166686373*pi) q[33];
U1q(0.170138560388493*pi,0.98812192666648*pi) q[34];
U1q(1.24525742311645*pi,1.9314245265043395*pi) q[35];
U1q(0.652797122212475*pi,1.9321009180933082*pi) q[36];
U1q(1.08092731877054*pi,1.766036662556004*pi) q[37];
U1q(0.513299859636652*pi,1.534656840316456*pi) q[38];
U1q(0.823922560340513*pi,1.031047559397872*pi) q[39];
RZZ(0.5*pi) q[31],q[0];
RZZ(0.5*pi) q[33],q[1];
RZZ(0.5*pi) q[35],q[2];
RZZ(0.5*pi) q[3],q[6];
RZZ(0.5*pi) q[4],q[24];
RZZ(0.5*pi) q[5],q[13];
RZZ(0.5*pi) q[39],q[7];
RZZ(0.5*pi) q[26],q[8];
RZZ(0.5*pi) q[9],q[28];
RZZ(0.5*pi) q[10],q[20];
RZZ(0.5*pi) q[37],q[11];
RZZ(0.5*pi) q[12],q[14];
RZZ(0.5*pi) q[19],q[15];
RZZ(0.5*pi) q[16],q[25];
RZZ(0.5*pi) q[22],q[17];
RZZ(0.5*pi) q[18],q[27];
RZZ(0.5*pi) q[29],q[21];
RZZ(0.5*pi) q[23],q[32];
RZZ(0.5*pi) q[34],q[30];
RZZ(0.5*pi) q[38],q[36];
U1q(0.106463929471562*pi,1.766457589870683*pi) q[0];
U1q(0.410092974007126*pi,1.44561628038137*pi) q[1];
U1q(3.635411710828253*pi,0.9740646158480493*pi) q[2];
U1q(1.45096780059741*pi,1.36781788687522*pi) q[3];
U1q(1.21363365530894*pi,1.86715201148295*pi) q[4];
U1q(3.656560757420369*pi,1.78297473140241*pi) q[5];
U1q(3.248939918556761*pi,0.8616307210905616*pi) q[6];
U1q(1.41422710460303*pi,1.3395665121975902*pi) q[7];
U1q(3.440905295495209*pi,0.7443632752157324*pi) q[8];
U1q(1.1094245442685*pi,1.2802011368648805*pi) q[9];
U1q(3.582221349695077*pi,0.42369271896056726*pi) q[10];
U1q(3.762331294005982*pi,1.986488609495451*pi) q[11];
U1q(0.796877946283995*pi,1.39610704749921*pi) q[12];
U1q(3.459319681890424*pi,0.09829977779602411*pi) q[13];
U1q(1.71740443259852*pi,0.451469157797173*pi) q[14];
U1q(1.1335878319941*pi,1.18200177891493*pi) q[15];
U1q(0.48396020827398*pi,0.28521435297659004*pi) q[16];
U1q(1.7186782097235*pi,0.5324005513612952*pi) q[17];
U1q(3.893978142692543*pi,1.1046430913753045*pi) q[18];
U1q(3.3199115244117747*pi,1.3557064969245807*pi) q[19];
U1q(0.381133829402488*pi,0.2863974004955201*pi) q[20];
U1q(3.384537842085221*pi,0.4002284191612495*pi) q[21];
U1q(0.674032803094244*pi,0.26721315716648997*pi) q[22];
U1q(0.248869437722238*pi,0.16196414020285*pi) q[23];
U1q(1.02827893927871*pi,0.8061573078952609*pi) q[24];
U1q(1.45974080395373*pi,1.7152068677356267*pi) q[25];
U1q(1.38563704356206*pi,1.4664825800735302*pi) q[26];
U1q(3.552911611008731*pi,0.806896658574719*pi) q[27];
U1q(1.36276316089659*pi,0.9190756968431304*pi) q[28];
U1q(3.869406293915641*pi,1.584730010847228*pi) q[29];
U1q(3.6570704539703778*pi,0.26036953811820995*pi) q[30];
U1q(3.741491949336735*pi,1.3381088165039539*pi) q[31];
U1q(3.037943797175685*pi,1.3711413186676846*pi) q[32];
U1q(1.74154957658035*pi,0.8879802009356152*pi) q[33];
U1q(0.402608310910171*pi,1.9308167148923698*pi) q[34];
U1q(3.241926515929591*pi,0.3830347473422201*pi) q[35];
U1q(1.19921637168591*pi,1.9414373653846302*pi) q[36];
U1q(3.557522794800529*pi,0.3397142445711344*pi) q[37];
U1q(1.49301696003719*pi,1.0186527477620002*pi) q[38];
U1q(0.105051613698539*pi,0.8424527713032002*pi) q[39];
RZZ(0.5*pi) q[0],q[35];
RZZ(0.5*pi) q[26],q[1];
RZZ(0.5*pi) q[38],q[2];
RZZ(0.5*pi) q[3],q[8];
RZZ(0.5*pi) q[25],q[4];
RZZ(0.5*pi) q[5],q[27];
RZZ(0.5*pi) q[6],q[36];
RZZ(0.5*pi) q[24],q[7];
RZZ(0.5*pi) q[9],q[32];
RZZ(0.5*pi) q[37],q[10];
RZZ(0.5*pi) q[33],q[11];
RZZ(0.5*pi) q[12],q[17];
RZZ(0.5*pi) q[13],q[31];
RZZ(0.5*pi) q[28],q[14];
RZZ(0.5*pi) q[20],q[15];
RZZ(0.5*pi) q[16],q[21];
RZZ(0.5*pi) q[18],q[30];
RZZ(0.5*pi) q[34],q[19];
RZZ(0.5*pi) q[22],q[39];
RZZ(0.5*pi) q[29],q[23];
U1q(0.787539638633116*pi,0.51517673946836*pi) q[0];
U1q(1.44000329192158*pi,1.03568451955692*pi) q[1];
U1q(1.74697569261868*pi,1.9112160884719227*pi) q[2];
U1q(3.520254845662053*pi,0.40021028683574755*pi) q[3];
U1q(3.432793939745519*pi,0.9787946062741597*pi) q[4];
U1q(1.40108736676641*pi,1.9475131714628766*pi) q[5];
U1q(1.68704621940287*pi,0.7991955666941066*pi) q[6];
U1q(3.387687090890651*pi,0.7687794747701604*pi) q[7];
U1q(3.070233488349496*pi,1.8501464443286157*pi) q[8];
U1q(0.516650180001799*pi,1.4484650216925954*pi) q[9];
U1q(3.614802357715848*pi,0.46159508743476607*pi) q[10];
U1q(1.39677140495906*pi,0.17651321866910186*pi) q[11];
U1q(0.465400135342593*pi,1.6416850840812698*pi) q[12];
U1q(1.37290556372183*pi,0.2735602321887645*pi) q[13];
U1q(1.13178455513552*pi,1.1184612521390465*pi) q[14];
U1q(1.46059315172744*pi,0.8585194871104829*pi) q[15];
U1q(3.7391000808709522*pi,1.3679476739600798*pi) q[16];
U1q(1.472931455209*pi,0.22409699278106532*pi) q[17];
U1q(1.87276646589793*pi,0.15267075446611078*pi) q[18];
U1q(3.0486640429744227*pi,1.6052464023840518*pi) q[19];
U1q(0.756681739836407*pi,0.18628658038967982*pi) q[20];
U1q(1.5469168010956*pi,1.2000783113643823*pi) q[21];
U1q(0.827855086088942*pi,1.25002407139096*pi) q[22];
U1q(0.225677145116131*pi,1.25087003707193*pi) q[23];
U1q(0.419997598533064*pi,0.4265151097646167*pi) q[24];
U1q(0.356012008461168*pi,1.4614677956968365*pi) q[25];
U1q(1.62694942305932*pi,0.7378160916899665*pi) q[26];
U1q(1.30930609374858*pi,0.4836694104559127*pi) q[27];
U1q(0.928823767711817*pi,1.0113414994562056*pi) q[28];
U1q(1.54284729518066*pi,0.9608404827666*pi) q[29];
U1q(1.55719376817736*pi,0.8797047554182598*pi) q[30];
U1q(3.23093614508654*pi,0.572463110113492*pi) q[31];
U1q(1.8603143027796*pi,0.7499162432885984*pi) q[32];
U1q(1.48272826638723*pi,1.596553596176955*pi) q[33];
U1q(0.667285751859603*pi,0.5842823564617499*pi) q[34];
U1q(1.35560418397662*pi,1.3408048705289974*pi) q[35];
U1q(3.438687370736745*pi,1.3383677741876787*pi) q[36];
U1q(3.6236825888564432*pi,0.8157891236114994*pi) q[37];
U1q(3.769522709515927*pi,1.5040896136871136*pi) q[38];
U1q(1.75749753950795*pi,0.8127741395146302*pi) q[39];
RZZ(0.5*pi) q[0],q[27];
RZZ(0.5*pi) q[1],q[39];
RZZ(0.5*pi) q[23],q[2];
RZZ(0.5*pi) q[3],q[30];
RZZ(0.5*pi) q[18],q[4];
RZZ(0.5*pi) q[5],q[6];
RZZ(0.5*pi) q[11],q[7];
RZZ(0.5*pi) q[37],q[8];
RZZ(0.5*pi) q[33],q[9];
RZZ(0.5*pi) q[12],q[10];
RZZ(0.5*pi) q[13],q[26];
RZZ(0.5*pi) q[20],q[14];
RZZ(0.5*pi) q[34],q[15];
RZZ(0.5*pi) q[16],q[31];
RZZ(0.5*pi) q[17],q[32];
RZZ(0.5*pi) q[38],q[19];
RZZ(0.5*pi) q[21],q[35];
RZZ(0.5*pi) q[22],q[36];
RZZ(0.5*pi) q[25],q[24];
RZZ(0.5*pi) q[28],q[29];
U1q(1.47406529482001*pi,0.7464075418897802*pi) q[0];
U1q(3.425622186914615*pi,1.9569674175860023*pi) q[1];
U1q(1.37345852974444*pi,0.514934442469678*pi) q[2];
U1q(3.645827299730951*pi,1.1592093209758576*pi) q[3];
U1q(1.51672842421988*pi,1.0174426271199708*pi) q[4];
U1q(0.507639839635108*pi,1.723146777280527*pi) q[5];
U1q(3.216258749681455*pi,0.0972788821778483*pi) q[6];
U1q(1.0927530879059*pi,0.39125807011043*pi) q[7];
U1q(3.488934600631522*pi,1.7617099025273855*pi) q[8];
U1q(0.628724292725209*pi,0.4910243030289704*pi) q[9];
U1q(3.134845535597511*pi,0.1070871357692662*pi) q[10];
U1q(0.632945479435573*pi,1.381789261443612*pi) q[11];
U1q(1.56214591717151*pi,0.27130251330972044*pi) q[12];
U1q(1.88000283356337*pi,1.0144823991331777*pi) q[13];
U1q(1.41483464950617*pi,1.7516004877150935*pi) q[14];
U1q(1.18345884705973*pi,1.2924187287536837*pi) q[15];
U1q(1.84961080585105*pi,0.9590850138817792*pi) q[16];
U1q(3.376038862179641*pi,0.541972509622917*pi) q[17];
U1q(1.61334814811453*pi,0.23350666347689053*pi) q[18];
U1q(3.486630164107784*pi,0.34834981890062466*pi) q[19];
U1q(1.67120758701182*pi,1.8427929837206198*pi) q[20];
U1q(0.536001000064195*pi,0.8208952324507521*pi) q[21];
U1q(0.785994215479756*pi,0.8307603288059999*pi) q[22];
U1q(1.17359880000018*pi,0.023946422838410086*pi) q[23];
U1q(0.460824447173799*pi,0.6220710247092311*pi) q[24];
U1q(0.353513132733405*pi,0.5088105924514565*pi) q[25];
U1q(0.330520414879871*pi,1.9580358994903566*pi) q[26];
U1q(0.871728286927765*pi,1.1100639787849724*pi) q[27];
U1q(0.49481507863108*pi,1.1789181329133904*pi) q[28];
U1q(0.666649590714049*pi,0.4897948965100409*pi) q[29];
U1q(1.54947693380361*pi,0.58838089025787*pi) q[30];
U1q(0.640755578337248*pi,1.3146363596394721*pi) q[31];
U1q(0.663524264654036*pi,1.1077848882239913*pi) q[32];
U1q(1.29504286055726*pi,0.01584708134757662*pi) q[33];
U1q(0.389284874625567*pi,0.5710551445072198*pi) q[34];
U1q(1.76290285105792*pi,0.16300606014389057*pi) q[35];
U1q(1.79284172354591*pi,0.2592753657824345*pi) q[36];
U1q(3.420792869572562*pi,0.44865622779279146*pi) q[37];
U1q(3.716874908220094*pi,1.9655274911058935*pi) q[38];
U1q(1.36070918965316*pi,1.1604439710649936*pi) q[39];
RZZ(0.5*pi) q[16],q[0];
RZZ(0.5*pi) q[1],q[8];
RZZ(0.5*pi) q[17],q[2];
RZZ(0.5*pi) q[3],q[14];
RZZ(0.5*pi) q[26],q[4];
RZZ(0.5*pi) q[5],q[37];
RZZ(0.5*pi) q[6],q[7];
RZZ(0.5*pi) q[31],q[9];
RZZ(0.5*pi) q[10],q[29];
RZZ(0.5*pi) q[38],q[11];
RZZ(0.5*pi) q[12],q[39];
RZZ(0.5*pi) q[13],q[28];
RZZ(0.5*pi) q[15],q[27];
RZZ(0.5*pi) q[18],q[21];
RZZ(0.5*pi) q[23],q[19];
RZZ(0.5*pi) q[20],q[30];
RZZ(0.5*pi) q[22],q[25];
RZZ(0.5*pi) q[34],q[24];
RZZ(0.5*pi) q[36],q[32];
RZZ(0.5*pi) q[33],q[35];
U1q(3.642495390583814*pi,1.131037909892842*pi) q[0];
U1q(1.87647594545166*pi,1.8325214937286596*pi) q[1];
U1q(3.879362256008925*pi,0.9738555169741288*pi) q[2];
U1q(1.46880644825867*pi,1.8794863091726475*pi) q[3];
U1q(0.916634029445668*pi,0.2767846031977139*pi) q[4];
U1q(0.320859834936461*pi,1.5039466806174264*pi) q[5];
U1q(1.45734939214354*pi,0.11506740541813798*pi) q[6];
U1q(1.39505659156553*pi,0.4600880527462303*pi) q[7];
U1q(3.224919765293275*pi,0.26814506106043146*pi) q[8];
U1q(1.39873912956316*pi,0.4332243107273124*pi) q[9];
U1q(1.40539185229482*pi,0.23415995576238613*pi) q[10];
U1q(0.591285095833847*pi,1.001476166749642*pi) q[11];
U1q(3.049813838576291*pi,0.944887692425997*pi) q[12];
U1q(1.43783320559281*pi,0.6097548899847982*pi) q[13];
U1q(1.13796101334021*pi,0.09433085353312398*pi) q[14];
U1q(1.32689716232051*pi,1.8436588673583199*pi) q[15];
U1q(1.74763938323602*pi,0.02695269801894984*pi) q[16];
U1q(0.59286171298959*pi,0.8413160638511998*pi) q[17];
U1q(1.32170400467371*pi,1.3348656947471387*pi) q[18];
U1q(3.819124371680268*pi,1.0938120756792657*pi) q[19];
U1q(3.397277722284645*pi,0.8243075132230295*pi) q[20];
U1q(0.461620165872279*pi,0.04377990475770305*pi) q[21];
U1q(0.608208310671334*pi,1.57100991852003*pi) q[22];
U1q(1.23349270242806*pi,1.436971103087779*pi) q[23];
U1q(0.629368919921577*pi,0.07195895474690062*pi) q[24];
U1q(1.3193223741198*pi,0.8021810832525067*pi) q[25];
U1q(3.329008099167084*pi,0.49770529940607666*pi) q[26];
U1q(1.53101817369399*pi,1.528269911943413*pi) q[27];
U1q(1.25562298269688*pi,0.7229645629663004*pi) q[28];
U1q(0.879688814512105*pi,1.7853622901945119*pi) q[29];
U1q(3.0989573901080227*pi,1.2017768317733348*pi) q[30];
U1q(1.11548247909863*pi,1.8026787545325016*pi) q[31];
U1q(1.42284065752672*pi,1.1909372271218412*pi) q[32];
U1q(0.514697122472278*pi,0.8038556643018362*pi) q[33];
U1q(0.468473609961474*pi,0.9657540204007402*pi) q[34];
U1q(1.8716563035478*pi,0.3607033919083882*pi) q[35];
U1q(0.379043257363075*pi,0.34724700630783456*pi) q[36];
U1q(3.332092739549428*pi,0.3898425502538434*pi) q[37];
U1q(1.64701299130325*pi,0.5038077268376076*pi) q[38];
U1q(1.48327510698862*pi,0.5882449574576336*pi) q[39];
RZZ(0.5*pi) q[1],q[0];
RZZ(0.5*pi) q[33],q[2];
RZZ(0.5*pi) q[3],q[35];
RZZ(0.5*pi) q[31],q[4];
RZZ(0.5*pi) q[5],q[23];
RZZ(0.5*pi) q[6],q[24];
RZZ(0.5*pi) q[32],q[7];
RZZ(0.5*pi) q[8],q[10];
RZZ(0.5*pi) q[9],q[14];
RZZ(0.5*pi) q[20],q[11];
RZZ(0.5*pi) q[12],q[18];
RZZ(0.5*pi) q[13],q[30];
RZZ(0.5*pi) q[29],q[15];
RZZ(0.5*pi) q[16],q[22];
RZZ(0.5*pi) q[38],q[17];
RZZ(0.5*pi) q[19],q[36];
RZZ(0.5*pi) q[28],q[21];
RZZ(0.5*pi) q[25],q[34];
RZZ(0.5*pi) q[37],q[26];
RZZ(0.5*pi) q[39],q[27];
U1q(3.658721536982891*pi,1.0117435359086926*pi) q[0];
U1q(3.53748531830868*pi,0.7780147407199554*pi) q[1];
U1q(3.31175896881099*pi,0.19580930236492478*pi) q[2];
U1q(1.41090476617353*pi,1.3740262520351676*pi) q[3];
U1q(1.32603493489916*pi,0.7656007678149439*pi) q[4];
U1q(1.27186627653664*pi,0.6100290287850463*pi) q[5];
U1q(0.327045107885552*pi,0.657002373065064*pi) q[6];
U1q(1.54000159859772*pi,1.6658857269575726*pi) q[7];
U1q(0.730845852290676*pi,1.7454346336338518*pi) q[8];
U1q(1.74859500505326*pi,0.1153998578981601*pi) q[9];
U1q(0.341967218466699*pi,1.6376003529923757*pi) q[10];
U1q(0.411823553841128*pi,0.3287168816922814*pi) q[11];
U1q(3.559837511487409*pi,0.36994471235609705*pi) q[12];
U1q(0.785261930903087*pi,1.8848145929124884*pi) q[13];
U1q(0.360091607152854*pi,1.1069848513714335*pi) q[14];
U1q(1.19875126706563*pi,0.1252493435301396*pi) q[15];
U1q(1.67413425855977*pi,0.25639151874572086*pi) q[16];
U1q(0.821031704575141*pi,1.068512837102337*pi) q[17];
U1q(0.674652925137119*pi,0.33237482506557825*pi) q[18];
U1q(1.5258568935787*pi,0.21280093486587748*pi) q[19];
U1q(0.518532061895211*pi,0.08697729267782961*pi) q[20];
U1q(1.65720411844726*pi,0.6405471526970321*pi) q[21];
U1q(1.53519236913344*pi,1.04991695355973*pi) q[22];
U1q(0.595610893452803*pi,1.0020143278582179*pi) q[23];
U1q(0.522559661748754*pi,0.6522932765402203*pi) q[24];
U1q(3.337534817796841*pi,1.9423688823093768*pi) q[25];
U1q(3.41384833894162*pi,1.7687057248941365*pi) q[26];
U1q(1.79899575972824*pi,1.1219068656539815*pi) q[27];
U1q(3.5828378793809392*pi,0.8689234310296232*pi) q[28];
U1q(1.71534930298781*pi,1.3237976741483548*pi) q[29];
U1q(3.326872953297471*pi,0.6514442631450748*pi) q[30];
U1q(3.32858662098149*pi,1.0033358473842044*pi) q[31];
U1q(3.9126793378243536*pi,1.774398350903855*pi) q[32];
U1q(1.34348899190918*pi,0.5394572948237268*pi) q[33];
U1q(0.808075073571473*pi,0.22808595551039978*pi) q[34];
U1q(0.590288633953656*pi,0.5600513846302579*pi) q[35];
U1q(1.89911914467109*pi,1.2824731530649949*pi) q[36];
U1q(3.4034320190550043*pi,1.747378964909204*pi) q[37];
U1q(0.537150652442785*pi,1.1158639438522076*pi) q[38];
U1q(3.866599004644633*pi,0.2969679100280489*pi) q[39];
RZZ(0.5*pi) q[0],q[10];
RZZ(0.5*pi) q[13],q[1];
RZZ(0.5*pi) q[9],q[2];
RZZ(0.5*pi) q[5],q[3];
RZZ(0.5*pi) q[21],q[4];
RZZ(0.5*pi) q[6],q[35];
RZZ(0.5*pi) q[22],q[7];
RZZ(0.5*pi) q[8],q[29];
RZZ(0.5*pi) q[11],q[36];
RZZ(0.5*pi) q[12],q[20];
RZZ(0.5*pi) q[34],q[14];
RZZ(0.5*pi) q[24],q[15];
RZZ(0.5*pi) q[16],q[17];
RZZ(0.5*pi) q[18],q[32];
RZZ(0.5*pi) q[25],q[19];
RZZ(0.5*pi) q[30],q[23];
RZZ(0.5*pi) q[26],q[27];
RZZ(0.5*pi) q[37],q[28];
RZZ(0.5*pi) q[31],q[39];
RZZ(0.5*pi) q[33],q[38];
U1q(1.78781173028699*pi,0.658367616084484*pi) q[0];
U1q(1.53304530656073*pi,0.5540527295226472*pi) q[1];
U1q(1.40130645169329*pi,1.4351661346104319*pi) q[2];
U1q(1.3991334810742*pi,0.7436841567818879*pi) q[3];
U1q(1.86887557545426*pi,0.9768119163479816*pi) q[4];
U1q(3.6328728610033822*pi,0.865404247164216*pi) q[5];
U1q(0.506091298252687*pi,0.8592452678573439*pi) q[6];
U1q(0.494353744701168*pi,0.6814725606294827*pi) q[7];
U1q(0.609895534740445*pi,0.3546949073992437*pi) q[8];
U1q(0.45434661734571*pi,0.4664238348052603*pi) q[9];
U1q(0.702385324670674*pi,1.131308324834876*pi) q[10];
U1q(0.638023924490934*pi,1.8749575232650821*pi) q[11];
U1q(1.21941009518572*pi,0.4655386111895061*pi) q[12];
U1q(0.655979386517718*pi,0.2727429636840082*pi) q[13];
U1q(0.72183909648314*pi,0.724272425647464*pi) q[14];
U1q(3.373673196568628*pi,0.17147980116672912*pi) q[15];
U1q(0.484806549502363*pi,0.5705008622417722*pi) q[16];
U1q(0.416347298202812*pi,1.8755854209070169*pi) q[17];
U1q(0.718787660113722*pi,0.3048537046703883*pi) q[18];
U1q(0.504623166629847*pi,0.9493111263797704*pi) q[19];
U1q(0.519077040986467*pi,0.9806611848236084*pi) q[20];
U1q(3.403262225074962*pi,0.8746497640546904*pi) q[21];
U1q(1.73951299541262*pi,1.2972992164336246*pi) q[22];
U1q(0.722914148795713*pi,0.41914811391436757*pi) q[23];
U1q(0.625445600694927*pi,1.3550349844985208*pi) q[24];
U1q(1.55994958460356*pi,0.9283873473341366*pi) q[25];
U1q(0.83531645036611*pi,0.1184669464739958*pi) q[26];
U1q(0.68326633912479*pi,0.3521563540907211*pi) q[27];
U1q(1.70927549273538*pi,1.207836782034783*pi) q[28];
U1q(1.65864741431773*pi,1.8310070245434822*pi) q[29];
U1q(3.862207715710371*pi,0.5102472585971327*pi) q[30];
U1q(0.281089050467559*pi,1.3352691462717647*pi) q[31];
U1q(1.74066481961448*pi,0.6690905635142474*pi) q[32];
U1q(1.15204225148832*pi,0.533254505929456*pi) q[33];
U1q(0.766150107552658*pi,1.6923333744003006*pi) q[34];
U1q(0.671494676920924*pi,0.752140862285088*pi) q[35];
U1q(3.100074203151483*pi,0.028171922127356197*pi) q[36];
U1q(0.111370497341832*pi,0.07533049423813698*pi) q[37];
U1q(0.601512434990468*pi,0.47656526600086746*pi) q[38];
U1q(1.35032829297963*pi,0.027291287273392584*pi) q[39];
rz(1.341632383915516*pi) q[0];
rz(1.4459472704773528*pi) q[1];
rz(0.5648338653895681*pi) q[2];
rz(1.256315843218112*pi) q[3];
rz(3.0231880836520184*pi) q[4];
rz(1.134595752835784*pi) q[5];
rz(3.140754732142656*pi) q[6];
rz(1.3185274393705173*pi) q[7];
rz(3.6453050926007564*pi) q[8];
rz(3.5335761651947397*pi) q[9];
rz(2.868691675165124*pi) q[10];
rz(2.125042476734918*pi) q[11];
rz(3.534461388810494*pi) q[12];
rz(1.7272570363159918*pi) q[13];
rz(1.275727574352536*pi) q[14];
rz(3.828520198833271*pi) q[15];
rz(3.429499137758228*pi) q[16];
rz(0.12441457909298315*pi) q[17];
rz(1.6951462953296117*pi) q[18];
rz(1.0506888736202296*pi) q[19];
rz(3.0193388151763916*pi) q[20];
rz(3.1253502359453096*pi) q[21];
rz(0.7027007835663754*pi) q[22];
rz(3.5808518860856324*pi) q[23];
rz(2.644965015501479*pi) q[24];
rz(1.0716126526658634*pi) q[25];
rz(1.8815330535260042*pi) q[26];
rz(1.647843645909279*pi) q[27];
rz(0.7921632179652169*pi) q[28];
rz(0.1689929754565178*pi) q[29];
rz(3.4897527414028673*pi) q[30];
rz(0.6647308537282353*pi) q[31];
rz(3.3309094364857526*pi) q[32];
rz(1.466745494070544*pi) q[33];
rz(2.3076666255996994*pi) q[34];
rz(3.247859137714912*pi) q[35];
rz(3.971828077872644*pi) q[36];
rz(1.924669505761863*pi) q[37];
rz(3.5234347339991325*pi) q[38];
rz(1.9727087127266074*pi) q[39];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];
measure q[5] -> c[5];
measure q[6] -> c[6];
measure q[7] -> c[7];
measure q[8] -> c[8];
measure q[9] -> c[9];
measure q[10] -> c[10];
measure q[11] -> c[11];
measure q[12] -> c[12];
measure q[13] -> c[13];
measure q[14] -> c[14];
measure q[15] -> c[15];
measure q[16] -> c[16];
measure q[17] -> c[17];
measure q[18] -> c[18];
measure q[19] -> c[19];
measure q[20] -> c[20];
measure q[21] -> c[21];
measure q[22] -> c[22];
measure q[23] -> c[23];
measure q[24] -> c[24];
measure q[25] -> c[25];
measure q[26] -> c[26];
measure q[27] -> c[27];
measure q[28] -> c[28];
measure q[29] -> c[29];
measure q[30] -> c[30];
measure q[31] -> c[31];
measure q[32] -> c[32];
measure q[33] -> c[33];
measure q[34] -> c[34];
measure q[35] -> c[35];
measure q[36] -> c[36];
measure q[37] -> c[37];
measure q[38] -> c[38];
measure q[39] -> c[39];