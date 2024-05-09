OPENQASM 2.0;
include "hqslib1.inc";

qreg q[16];
creg c[16];
rz(3.8518872365927894*pi) q[0];
rz(2.5144333612200214*pi) q[1];
rz(3.207355270434757*pi) q[2];
rz(0.0688235113834641*pi) q[3];
rz(0.5447392242678616*pi) q[4];
rz(0.45466389444141975*pi) q[5];
rz(1.2981036620079378*pi) q[6];
rz(1.7065448660727824*pi) q[7];
rz(1.36103838669792*pi) q[8];
rz(3.694543187576875*pi) q[9];
rz(1.86539962088079*pi) q[10];
rz(3.1886400485099644*pi) q[11];
rz(3.270898880417677*pi) q[12];
rz(1.0001152283846273*pi) q[13];
rz(3.862638793975498*pi) q[14];
rz(0.3201409791894135*pi) q[15];
U1q(1.83939594315985*pi,0.575134558748694*pi) q[0];
U1q(3.764928910708531*pi,1.477654293744942*pi) q[1];
U1q(3.824685276403319*pi,1.9208220062701498*pi) q[2];
U1q(0.527073964137442*pi,1.747608305844951*pi) q[3];
U1q(3.421528651617871*pi,1.815360316366167*pi) q[4];
U1q(1.68615165212298*pi,0.677487511771448*pi) q[5];
U1q(1.31521545571772*pi,0.907133234897161*pi) q[6];
U1q(1.1851115821933*pi,1.17345366839617*pi) q[7];
U1q(0.628193264051339*pi,0.694232731304576*pi) q[8];
U1q(0.218922533360279*pi,1.323813381474175*pi) q[9];
U1q(0.844362538952572*pi,1.4718078665593*pi) q[10];
U1q(1.39541759605901*pi,0.163611256178182*pi) q[11];
U1q(0.747751300973735*pi,1.858167105250879*pi) q[12];
U1q(1.62825677603399*pi,0.697381204095291*pi) q[13];
U1q(0.436331926077044*pi,1.9473446720997634*pi) q[14];
U1q(1.81247142322463*pi,0.0133101518397942*pi) q[15];
RZZ(0.0*pi) q[9],q[0];
RZZ(0.0*pi) q[4],q[1];
RZZ(0.0*pi) q[3],q[2];
RZZ(0.0*pi) q[5],q[13];
RZZ(0.0*pi) q[6],q[11];
RZZ(0.0*pi) q[7],q[14];
RZZ(0.0*pi) q[8],q[10];
RZZ(0.0*pi) q[12],q[15];
rz(3.356765661526616*pi) q[0];
rz(0.22384097130784*pi) q[1];
rz(0.124183444386374*pi) q[2];
rz(0.708389050604524*pi) q[3];
rz(1.33873296853226*pi) q[4];
rz(2.19193265910412*pi) q[5];
rz(3.538131178267625*pi) q[6];
rz(3.768654428140578*pi) q[7];
rz(3.782847474252648*pi) q[8];
rz(3.2950514565570073*pi) q[9];
rz(3.226535953229567*pi) q[10];
rz(1.03339357566335*pi) q[11];
rz(3.821041358851933*pi) q[12];
rz(1.12422942310816*pi) q[13];
rz(3.722507787245538*pi) q[14];
rz(0.490938946064742*pi) q[15];
U1q(0.695519494580385*pi,0.178756021111761*pi) q[0];
U1q(0.333359808561166*pi,1.205784800017373*pi) q[1];
U1q(0.85079660918106*pi,1.9758738173980097*pi) q[2];
U1q(0.48807994228035*pi,0.887952528778201*pi) q[3];
U1q(0.693295662608699*pi,1.38688837231418*pi) q[4];
U1q(0.716969879236548*pi,1.148039370290445*pi) q[5];
U1q(0.408494883396287*pi,1.1257304897835199*pi) q[6];
U1q(0.677509553983826*pi,0.241036390102022*pi) q[7];
U1q(0.280583978305661*pi,1.186359339288292*pi) q[8];
U1q(0.740285479693881*pi,1.563399303076139*pi) q[9];
U1q(0.713962101013856*pi,1.832307266968772*pi) q[10];
U1q(0.458412580069868*pi,1.924951835954512*pi) q[11];
U1q(0.523736721697016*pi,1.604740351320894*pi) q[12];
U1q(0.380611673432221*pi,0.822579759027527*pi) q[13];
U1q(0.810899756857444*pi,1.765186056132838*pi) q[14];
U1q(0.159014086872305*pi,1.3159888765015801*pi) q[15];
RZZ(0.0*pi) q[5],q[0];
RZZ(0.0*pi) q[1],q[2];
RZZ(0.0*pi) q[3],q[11];
RZZ(0.0*pi) q[4],q[9];
RZZ(0.0*pi) q[6],q[14];
RZZ(0.0*pi) q[7],q[15];
RZZ(0.0*pi) q[12],q[8];
RZZ(0.0*pi) q[13],q[10];
rz(3.895875955862044*pi) q[0];
rz(0.859769048334882*pi) q[1];
rz(2.28921192202903*pi) q[2];
rz(2.3620205266166*pi) q[3];
rz(2.42150323150899*pi) q[4];
rz(0.155435377092635*pi) q[5];
rz(1.21926318364871*pi) q[6];
rz(0.145751359076373*pi) q[7];
rz(3.9868547212215146*pi) q[8];
rz(2.55645369848388*pi) q[9];
rz(3.416172316945124*pi) q[10];
rz(1.75522422489851*pi) q[11];
rz(3.68619878181527*pi) q[12];
rz(1.18983717370159*pi) q[13];
rz(1.37902562289376*pi) q[14];
rz(0.401242213376564*pi) q[15];
U1q(0.311485412152416*pi,1.3618268345091789*pi) q[0];
U1q(0.404770587844845*pi,0.444061166263103*pi) q[1];
U1q(0.567264511889393*pi,1.377903523051162*pi) q[2];
U1q(0.713513811720625*pi,1.695794645390092*pi) q[3];
U1q(0.695382345035176*pi,1.9058733351185504*pi) q[4];
U1q(0.677753586201106*pi,1.828318096219131*pi) q[5];
U1q(0.569276778949003*pi,0.845143974935173*pi) q[6];
U1q(0.57264689641663*pi,1.9745367635554294*pi) q[7];
U1q(0.259594026929401*pi,1.9826491817568104*pi) q[8];
U1q(0.612877789010872*pi,1.423709033313542*pi) q[9];
U1q(0.624643424082824*pi,1.9209593788926362*pi) q[10];
U1q(0.571633376454148*pi,1.04792419107996*pi) q[11];
U1q(0.775054235326204*pi,1.691833706050715*pi) q[12];
U1q(0.54794386151095*pi,1.36400240437635*pi) q[13];
U1q(0.331726691733157*pi,1.810471677745766*pi) q[14];
U1q(0.85312093136077*pi,0.22248797227288*pi) q[15];
RZZ(0.0*pi) q[0],q[15];
RZZ(0.0*pi) q[1],q[14];
RZZ(0.0*pi) q[6],q[2];
RZZ(0.0*pi) q[3],q[7];
RZZ(0.0*pi) q[4],q[11];
RZZ(0.0*pi) q[5],q[8];
RZZ(0.0*pi) q[9],q[13];
RZZ(0.0*pi) q[12],q[10];
rz(1.25048866571777*pi) q[0];
rz(0.823352611483043*pi) q[1];
rz(0.519070358342735*pi) q[2];
rz(1.24095651021656*pi) q[3];
rz(0.111347963067428*pi) q[4];
rz(3.70013981729125*pi) q[5];
rz(3.760637122663254*pi) q[6];
rz(0.878132139673629*pi) q[7];
rz(1.34921045690348*pi) q[8];
rz(1.06918519973683*pi) q[9];
rz(0.466540270482264*pi) q[10];
rz(1.42823754617935*pi) q[11];
rz(3.298687608195139*pi) q[12];
rz(3.785406272436211*pi) q[13];
rz(1.21748618354163*pi) q[14];
rz(0.74655167594195*pi) q[15];
U1q(0.34814391109213*pi,0.721198236074828*pi) q[0];
U1q(0.474469018214378*pi,1.861630589037695*pi) q[1];
U1q(0.616936304915576*pi,0.913723859810853*pi) q[2];
U1q(0.44084977637402*pi,0.488574881756605*pi) q[3];
U1q(0.546281464245594*pi,0.361143633723241*pi) q[4];
U1q(0.443502829695873*pi,0.372312169517298*pi) q[5];
U1q(0.491828302757743*pi,0.609169241292089*pi) q[6];
U1q(0.254443076178174*pi,1.6605785507261581*pi) q[7];
U1q(0.423391030662775*pi,0.277953518397237*pi) q[8];
U1q(0.394807273017325*pi,0.502780958239819*pi) q[9];
U1q(0.696645349962962*pi,0.269102284238632*pi) q[10];
U1q(0.651974984609189*pi,0.682208192392821*pi) q[11];
U1q(0.641657241277174*pi,0.185440887733426*pi) q[12];
U1q(0.280109856670124*pi,0.842081791883845*pi) q[13];
U1q(0.75131749587611*pi,0.727629037249254*pi) q[14];
U1q(0.48068771137116*pi,0.870805859051615*pi) q[15];
RZZ(0.0*pi) q[0],q[10];
RZZ(0.0*pi) q[1],q[6];
RZZ(0.0*pi) q[8],q[2];
RZZ(0.0*pi) q[3],q[14];
RZZ(0.0*pi) q[4],q[12];
RZZ(0.0*pi) q[5],q[11];
RZZ(0.0*pi) q[7],q[13];
RZZ(0.0*pi) q[9],q[15];
rz(2.6154124442538897*pi) q[0];
rz(2.31897804288351*pi) q[1];
rz(0.53983848408574*pi) q[2];
rz(2.12541288899855*pi) q[3];
rz(2.6035372493337903*pi) q[4];
rz(0.237986028829149*pi) q[5];
rz(3.54033481427451*pi) q[6];
rz(2.02260143468548*pi) q[7];
rz(1.93171070000795*pi) q[8];
rz(2.74830828904843*pi) q[9];
rz(1.18723581228797*pi) q[10];
rz(0.587880627638248*pi) q[11];
rz(1.7399999102877102*pi) q[12];
rz(3.513693007661664*pi) q[13];
rz(0.895920531706986*pi) q[14];
rz(3.877270501761719*pi) q[15];
U1q(3.273904031405756*pi,0.70686509051807*pi) q[0];
U1q(3.134548881083373*pi,1.18799362835488*pi) q[1];
U1q(3.712627840112928*pi,0.275797948190114*pi) q[2];
U1q(3.503012945592278*pi,0.750716525584807*pi) q[3];
U1q(3.684456452935267*pi,0.9895887791419999*pi) q[4];
U1q(3.478567026329279*pi,0.294868933681616*pi) q[5];
U1q(3.250624062687004*pi,1.95527524412265*pi) q[6];
U1q(3.399107441908055*pi,1.40493225113105*pi) q[7];
U1q(3.520560576019811*pi,1.71392048022304*pi) q[8];
U1q(3.511716183305646*pi,1.58670517373623*pi) q[9];
U1q(3.490014121429979*pi,0.293129640788728*pi) q[10];
U1q(3.395963665190093*pi,0.238987250529967*pi) q[11];
U1q(3.469053050272342*pi,1.6096196193330141*pi) q[12];
U1q(3.0993491751941162*pi,0.559069995831818*pi) q[13];
U1q(3.885509508334647*pi,1.6003639644021699*pi) q[14];
U1q(3.578338733579924*pi,1.0465099028888711*pi) q[15];
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