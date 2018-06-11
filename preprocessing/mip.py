import io
import logging

# import mayavi.mlab
import numpy as np
# import pandas as pd
from PIL import Image
from google.cloud import storage
from scipy import misc
from tensorflow.python.lib.io import file_io
from matplotlib import pyplot as plt


def configure_logger():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


TRAINING_LIST = ['P4AIB8JMDY6RDRAP.npy', 'ASD2URNKFRN3ZSFL.npy',
                 'SRKOPGCEG62ZJTT2.npy', 'N7C279O07IXSUAX9.npy',
                 'UZFAM45KRI3C7MGB.npy', 'JMWZRLKLB5DSSXE3.npy',
                 'ANLMZN6NS4S3TCCF.npy', 'DSIFGAA3BMQKFE7A.npy',
                 'DDWAA4OECH5TXONN.npy', 'TSZFE43KG3NQJR69.npy',
                 'TFCLOHN4ROEYF6QY.npy', 'RKXO5JUDKMYRUTN5.npy',
                 'FF0O9FGXXC1W2G1N.npy', 'W6OUXVVFEFZSLKWT.npy',
                 'OLYVZJAPCZJQWGIE.npy', 'NJDN23E34MZVQU66.npy',
                 'NTFYWO8XKLEW332D.npy', 'KRWJSE7TVM2QLJI7.npy',
                 '6JWAB7EKJUL10ACK.npy', 'WW6NZ08FGRHMYND4.npy',
                 'B63PZNF524OFKIZT.npy', 'HXKDQ5I26Z2CRUQB.npy',
                 'RPR59VBOFAUEQ7BQ.npy', 'MKIAFF7LDGXTZN4C.npy',
                 'RWYZUDIPTYFCZ6UJ.npy', 'KS626OB4OI9MWE8F.npy',
                 'YEP4A7J1PJN5BMDO.npy', 'JAS2TYJWKQL65CQT.npy',
                 'K3C28ASFAF7ZLEIH.npy', '3BGIK8ODI5L6GCW4.npy',
                 'TVY4XBCLXJBKK72S.npy', 'DZLYYZNR6M5BZFBT.npy',
                 '0KSBX96F8BU1FCFQ.npy', 'YDJX25T27CYATQ25.npy',
                 'VKDRQPXRMRRBAHW3.npy', 'OQWBV4PPVHOOYRWI.npy',
                 'ZXXMMCGK6ANRKLFD.npy', 'EG36PUNTKBPUZ4PG.npy',
                 '21A1B91HWFSGQOAU.npy', 'JGNEVHX7BEKLBQJY.npy',
                 'ZMP6NER3I4ERS2JE.npy', 'EQD7NRTYLRGYENOM.npy',
                 'ENBHS75LQ7FWO5CP.npy', 'JTGK554CWBB6JDQJ.npy',
                 'XO2Z2XFH26YISRUA.npy', 'D8C02RJJ2Z63XMFD.npy',
                 'NXDFBWPXYWHEIO2O.npy', 'XKHDNXXCRND47JLF.npy',
                 'XQBRGW3CYGNUMWHI.npy', 'ZC9H37RWIQ90483S.npy',
                 '7GRXTJUB2643Z413.npy', '8S0FHQGBA5LPLI4I.npy',
                 'RFSGSDQ5MWIVM6QV.npy', '9Z9TDQNM92R1FK71.npy',
                 'CFC7SE7TGP55RMUB.npy', 'FLSUE7ZIXI6HSJWN.npy',
                 'H9PROZNCEP1LMPEG.npy', 'NJNFVZMSGBGM6H5Y.npy',
                 'ECJB3HH5WPIVOJJW.npy', 'CJ8TW38Y0NL2YDAR.npy',
                 'NIED5UTOZ2JOFCWT.npy', 'Q2ZSL21Z90QTPXY5.npy',
                 'XYAFZGBXWKQJN365.npy', 'YRC3D4LSQNDZCXDA.npy',
                 'JLHG7VESTU3GH2KX.npy', 'LEE0EERR3GTGT27L.npy',
                 'RRZGAIOEIS6WPXDT.npy', '5LGDWR05UU6NKX4Z.npy',
                 'ZTGZBI2QX2JQFXIG.npy', 'OOAKLUZ2A4XYXC4X.npy',
                 'NXLFQLVZRLUEK2UF.npy', 'W098UN5Y50II02BX.npy',
                 'M0896BIM3W9DCUU6.npy', 'TNDI8Z4QHMRPT40Q.npy',
                 'SQD3UDNZH2Q3ET4V.npy', 'NJRQL9A1NKXR0I0K.npy',
                 'IIMERRCV5YJD0ZXZ.npy', 'AFEB6XW5UCQTSYJL.npy',
                 'I2XXOTQBYCGCYHXS.npy', 'O3L9K24EI9IVZKAN.npy',
                 'ZSOO1JLX7GWD1AXI.npy', 'UGXVSPJLHJL6AHSW.npy',
                 'HIA2VPHI6ABMCQTV.npy', 'DVYKX4XMBHD57MZJ.npy',
                 'LUVMEPI5JWYL67RF.npy', 'PYRDN2YA3MWCWKCC.npy',
                 'DZD32LMZUMW2VBIH.npy', 'RKFD4L7QSBXPHFJU.npy',
                 'FYSSUUSC5XYNLUQW.npy', 'C5LCU8BQOA9HKWM6.npy',
                 'PNPGGIFOXWJJQ22P.npy', 'OQNPNGEBR4D54PRA.npy',
                 'ROMHUDLTCOEXUQCA.npy', 'VKFW6CO5FV2P3PUE.npy',
                 'WUG3VLS7FYXG35LH.npy', 'ZYKNNBEIY0387GUQ.npy',
                 'OAEVXX4IFAX5IYOR.npy', '6XB4FHWAXDVQ79NK.npy',
                 'UGAYD4C2EMSUZ7LV.npy', 'GKDV3FW4M56I3IKV.npy',
                 '75DQ5DHR0734PLWN.npy', 'WIJONRZUXV2SJOMN.npy',
                 'TXOGS5CQ6RZMSV3W.npy', 'RLZKSYKP7FOKRC1O.npy',
                 'OPLQZBP3BOBS4LTY.npy', '760N11QRA3L7166Z.npy',
                 '2B2XLE17Y0C5WJ3T.npy', 'FHUAUDALQMAQK3K8.npy',
                 '99YJX0CY4FHHW46S.npy', 'IGIIMA3LABMNETMA.npy',
                 'MLHE2CESZAWQKZVP.npy', 'AQCA8X3KQ99EFUQK.npy',
                 'LQMW3GK5JQEVVFBK.npy', 'MSZB67BPZ32ILNLK.npy',
                 'AOY7EXEF3H62XKG4.npy', 'MKBXWA7XQVFZRVRK.npy',
                 'EH3P3JW1V3L8GNJ7.npy', 'PGB3UF6PM780GAIE.npy',
                 'TUX27MONFI7VC6M3.npy', 'TBNR4PRU7XIL27G6.npy',
                 '35KSO21226Q8WCQA.npy', 'NWRLBSNF3RVU3MAC.npy',
                 'PFIIEXEJM7FY3N5Q.npy', 'YOZ19SX2BIALZHYO.npy',
                 'IF41GHNSIEOP7BWD.npy', 'S1MFJS5SSJ12LGEF.npy',
                 '86HEMQMXXRRXM9S0.npy', 'WIPV3EQ34O55HCUI.npy',
                 'STCSWQHX4UN23CDK.npy', 'MRYGCR7462Z7IN2L.npy',
                 'MSXY2JKZXPEBXR72.npy', 'LCGP6HURGG2XVJ5B.npy',
                 'WWEFFBIMLZ3KLQVZ.npy', '27QV9Z81JZB09QPH.npy',
                 'TFV4472GKPYD269N.npy', '4TM6IQICWY502GS3.npy',
                 'KHXSOC18T31CFB18.npy', 'CSQRNTS8K61O7WP5.npy',
                 'PQLRGWB8W5XDIP1B.npy', 'AFN5EQGSDERZCTLM.npy',
                 'WHCQX5APWQZHMM3Q.npy', 'EXX265Z1JC762YB7.npy',
                 'GIPUIOSNW2C34HIS.npy', 'HID1X1SOO57L1A87.npy',
                 'DJGKDC32BE5NKELX.npy', 'CSFRT7ES11I1Y3CF.npy',
                 'YLLD4EJGLL59WBG6.npy', 'JNTL5ESEIA5FWMUQ.npy',
                 'OAYNHJ5ZLLVYHXCA.npy', 'IDUHFQWVS4WKXPC5.npy',
                 'CF7JC6FO4N0S96KN.npy', '0MTDDGCF20DKOR59.npy',
                 'CWWKNDYG2UERXH7X.npy', 'XXEE333ZVHOJTRJJ.npy',
                 'CQHAXTV5WJLZBRFB.npy', 'MYB72GJF0W90NZWH.npy',
                 'IAR66U7QVOBXOFQE.npy', 'MD1ZI5BSP1QU1CUR.npy',
                 'CIED0H64G5Z7173O.npy', 'TGXRI16H1J321FI0.npy',
                 'CUM2CZJVM5H5ZKFE.npy', 'Z79AE6B47YSGBITL.npy',
                 'LZBLW6F81ERF8901.npy', 'PCNMFAZL5VWWK7RP.npy',
                 'ECLDQAP52JUWQZHM.npy', 'NXSJBZWIQX5RKNO3.npy',
                 'HLXOSVDF27JWNCMJ.npy', 'OGRRL3DQAJGEL0ZH.npy',
                 'EYYSOUW6PR2WMBZS.npy', 'XWJSDQUY2US6L4NJ.npy',
                 'TEHE4SMNJTXDNW2O.npy', 'KOE9CU24WK2TUQ43.npy',
                 'BRSZTKDWTB2X5P3W.npy', 'MLQJEF1PHZHJ6PAZ.npy',
                 'K5640BXXD7YMIP8N.npy', 'FKHXWQEPTPIWXHBR.npy',
                 '8SREEDHBQF4ZGA02.npy', 'PWECQP8X5J7H4F2Y.npy',
                 'WKKADQ0Q6519IXWY.npy', 'R51O91ZJGN3T0FGI.npy',
                 'LT4EQ19PF206URYF.npy', 'QPKEZOB3X6B6HMCR.npy',
                 'WDA531I7IXEZY3RK.npy', 'JOJU5LPDDBBP6LYG.npy',
                 'DPSG75LCBZVBTRPO.npy', 'R79UN6BZQXMD6MXR.npy',
                 'HSKRXAPZ7WNBT2TI.npy', 'UMFGGCOJN6GDPQDV.npy',
                 'IXRSXXZI0S6L0EJI.npy', 'TS4UPSJ92SZ1EITM.npy',
                 'CSCIKXOMNAIB3LUQ.npy', '1H37S4G2MIXMQ3QS.npy',
                 'QKBS566N24RMCV5H.npy', 'E1VWJWG3V3QYSSWX.npy',
                 'CRKZOVFXYNXRGJYR.npy', 'ZBXZAYA6RXNBWB6C.npy',
                 '3AWM4ZZHCWJ8MREY.npy', 'LNU3P20QOML7YGMZ.npy',
                 'XDNEWPTKFLCWWUSV.npy', 'AOTGHQ6S5AD7LOBO.npy',
                 'VGF4YIHDAARLWS7A.npy', 'GG1EJGK0NHRCBIBA.npy',
                 'IETROE62UNT5QVWS.npy', 'JNKX5SWXMGUXPR07.npy',
                 'G06KIRHKXWJZV4JL.npy', '2A617MJ3LGPCXRTK.npy',
                 'DMYPNOEZ9IWGA8AV.npy', 'UQXCE0KMLR3C4058.npy',
                 'ANCCICBPHIDJ5PJ4.npy', '3LS918JYONAZ7EUN.npy',
                 'WYMX6QXJXTECQ81F.npy', 'Z9SQUSY4XPPSP4ND.npy',
                 'IP4X9W512RO56NQ7.npy', 'D4UFLMT22FYKZ9O3.npy',
                 'YRPLT2NPTAFJ25RR.npy', 'K32CUE69HPQBU2KG.npy',
                 'B2NIQIU0PWY0HANT.npy', 'GJ5CHFO5E33QWWNW.npy',
                 'P8D6HTN341UCLBIT.npy', 'GWGPR9W0KUGO4G6M.npy',
                 '6HI9ZYDOAX7Y6KKJ.npy', 'NCU63Q4O8N50O6ND.npy',
                 '3DMBZTLDT2LRD4LG.npy', '60BQLN9SD3IXJ5AI.npy',
                 'BIASJO43N3A7Z63W.npy', 'PMMV7MLZUWJTIMLR.npy',
                 'H0JM6Z3R5563HTJM.npy', '0E1AN2T001WORM02.npy',
                 'DRT3F4ITWPCCBB6T.npy', 'VRSBMSXUYCQ3NC5A.npy',
                 'FCYGZ75WMW6L4PJM.npy', '16TRHGYACZ8HYHUB.npy',
                 'LDS0IN2DPOSUBAX4.npy', 'QEMIBYDHSXNH2E6X.npy',
                 '74Q4CS2DM19P4CZ6.npy', 'C0BWKG4H07GUBILW.npy',
                 '1SDZGHSOKOCXJ82G.npy', 'DRS5DZGN5PVUZX4Y.npy',
                 'IBRRD5O4NGVPABDT.npy', '5D2RFZWIZ8G6K1HX.npy',
                 'NEYLUIYPCL6O74O6.npy', 'MYM4BYJZ8J0APLCG.npy',
                 'JCMIIGHKFDRU911D.npy', 'ZFMUMZVL6HFQCKSX.npy',
                 'KEYZSDHDH2DINCKO.npy', 'ZOH87VOPAWI557JV.npy',
                 'IWU76ETUE9UIZN8O.npy', 'TZES7PMHXV4XVJBV.npy',
                 'SBLQEZAZEXCC76C6.npy', 'VKMYOQD5A64UFEED.npy',
                 'OJFOT7U85BBO92CL.npy', '7W6NJWN9S0UF5A8N.npy',
                 'ZSXFDYQ4ST4Z6SRF.npy', 'OMC7W51BOE9ILIWF.npy',
                 'MKKOO7NDN26O6RLZ.npy', 'PKSQ49QA03C0GTJ8.npy',
                 '8MVXENAM66B4AWIF.npy', 'YRNJZUOBWLMG7ESZ.npy',
                 'F8S7F9FG1FR0R2AT.npy', 'PHE2B59NIQZC67UJ.npy',
                 'WI171X7KEOFWUIUO.npy', 'NLXOSUXOZZHPFQQX.npy',
                 'AKZ8T688ZRU9UTY2.npy', 'KZMPFXLAUCRM314K.npy',
                 'MHL54TPCTOQJ2I4K.npy', 'Z4AR5KSIXNI0BOR6.npy',
                 'JWVEP6ZLYCOE5IAE.npy', 'C1YSN4O8HKU7B7XG.npy',
                 'IJBSBL7IRLNRVXUC.npy', 'QI8CI3NDYYJKGP9M.npy',
                 'E0PAWC91DAO8VM1V.npy', 'X8NK5EEYCGJNLSF9.npy',
                 'RRZNNMZTLY4BJK7C.npy', 'OLIQUZ7ERFP4DJVD.npy',
                 'ZASCC2YE27A6IT45.npy', 'PQ918CDGQIFXXXN1.npy',
                 'LQG4BRZPSCMR5XAN.npy', 'GHVG2CNNRZ65UBEU.npy',
                 'KJAWMYSKQTYINTNC.npy', 'WAP327ZXF6FGV5E4.npy',
                 'M2FFKPT000DH1D3L.npy', 'EGIAP2GDM33PCW5A.npy',
                 'UZQJZEDGVTWFY23U.npy', 'VZOXMEUK7BN7PQH3.npy',
                 '53SO7XOB1AERHGDS.npy', '9AF84HK0K5CG471Y.npy',
                 'CMELUGEVB3ESBIAK.npy', 'J3N2GD4RC25OFS0Y.npy',
                 'CYIJD3WQ3J3Q67LF.npy', 'M67136GRDYSCB588.npy',
                 'PIORRASXKMO7LEXJ.npy', 'MBRJCXKG522HOWH7.npy',
                 'UZUCCRJIMUZMQSQE.npy', 'X1E3MA35PFPQXNB9.npy',
                 'SZTHO00BX3ZAE4Y5.npy', 'ZOROHVT5JZVIBEEX.npy',
                 'K0CJ00ZYHHJ5PXGA.npy', 'EPSAH2BCBGV7WQHM.npy',
                 'HH11LJ5K391B78ZG.npy', 'LOR7YOK3PYHXSHCM.npy',
                 'GXR5YMJB3PZQNLAC.npy', 'XSNEILWLDG2HVSVH.npy',
                 'LCS6AB1M7X4ZP854.npy', 'NJB6RYF62XWGMSE2.npy',
                 'YSEPTJX4TY1WQY1T.npy', 'SXF2FCCYEQLXMQO5.npy',
                 '0MG5EK1O64CPS887.npy', 'S6YJ44EEBN5T211U.npy',
                 'JUBFRXRHLG46KY42.npy', 'SRG1SXU9H044QXS2.npy',
                 'YP3O7PTO0AG3Q9LN.npy', 'BBKQNRFE6QHDDKTP.npy',
                 'VBPF3M5P5YIFVZ4L.npy', '2D2SR4DRRR4CCSQK.npy',
                 'FBJFYBF7A46A4GIM.npy', '8TLM2DUBYEE2GDH0.npy',
                 'PMLTPI058LNPOB6K.npy', 'S4C0WMVGSI601DMK.npy',
                 'IPFO2JH7ZZYTQHH2.npy', 'M1ZMVHI4X7C5J4OX.npy',
                 '6OWGYLAAJCBPWEZ4.npy', 'EOI25OVG7DHZ6KH2.npy',
                 '20180501202602580.npy', 'HO0Y1BBUIJE9IJIJ.npy',
                 '1IN3A0RW0UHOBHMA.npy', 'DD4MRFEOI10TRMOF.npy',
                 'WLNWYNU4G9124ILG.npy', 'GDYRQZ2185AOXLHC.npy',
                 'AXWF7YC1M4HM46D3.npy', 'HXLMZWH3SFX3SPAN.npy',
                 'F6MLW8PSZ0Z431D2.npy', 'Q7TZ7D9MDFNNTUIE.npy',
                 'XVPBCFNYTDI4HHW2.npy', 'VDL6YJQYER3CXFOI.npy',
                 'WZABX1OKJSMRLG6E.npy', 'WMMP0B4WE4H12UL5.npy',
                 'XJVLT6YE7FKB3T3B.npy', 'DDIKD2BEX8VMNNIQ.npy',
                 'D3YR1YUTS8O1PFNM.npy', 'SRKT7U91OM48G8XF.npy',
                 'IEJDGAS52VTH4G74.npy', 'BHYZMLL7GBZZLACW.npy',
                 'YBDQ08DHIDPA3J98.npy', '6BMRRS9RAZUPR3IL.npy',
                 'Y6Z334Z280BD5OYA.npy', '5HYJPO3D1N5PVO0C.npy',
                 'UMRXCXFIT4YDD7RQ.npy', 'WZPADD37RLXRX3IG.npy',
                 'QJ9YG0LHX1SFOKLN.npy', 'TQKQG8IN5G3017SZ.npy',
                 'LE9D7Q73ZSL4N528.npy', '20180501210438227.npy',
                 'JAANEAKBEZPKV4JV.npy', 'RLJSRNLTRUQAVT3L.npy',
                 'WIQMFFRDNGILW7XZ.npy', 'B3UZMY4KGFIXP8EU.npy',
                 '43ERS29KX3LPF334.npy', 'TFIG39JA77W6USD3.npy',
                 'UBDRB3DMSB6TJ76N.npy', 'JUOV5YWN6IS7OXIJ.npy',
                 '64V9V1DQO6O3JJ8F.npy', 'UJHKCBMXXLB5QHTP.npy',
                 'EYEDMZ3Z6OIKPERL.npy', 'OCP7EUZJEB7JUL65.npy',
                 'GORU15M7DI5O0W1I.npy', '8GW4B6QKOW3ALQCA.npy',
                 'MQ9P5PQ68PLOCT7O.npy', 'RNLVK5QDFK87IGDG.npy',
                 'KQSTLFNPLNP5M43F.npy', 'JID2LWYOKIWCMT2Z.npy',
                 'FUVL4PLSUHRCEAHC.npy', 'SMGWMDYTYR8ZB3F5.npy',
                 'HGNXVLY1NGPI0B3T.npy', 'TCD5M6UQD4FBL6NM.npy',
                 '04IOS24JP70LHBGB.npy', 'M66OIOML76PWORZP.npy',
                 'GCBJDIT55TRGHZIR.npy', '1BBPJ3U25RYGSRSS.npy',
                 'TVCP433TKOVBCQTW.npy', 'AUGXT5PI4VY62M6L.npy',
                 'RKBSU42WA7AY22E7.npy', 'H72HW588HYXAZTGI.npy',
                 '0R1R98REO7SARJ06.npy', 'VOPMJMGIDQQ75E6G.npy',
                 'BMFSUHVBPJ7RY56P.npy', 'ABPO2BORDNF3OVL3.npy',
                 'VZHXKDEYF7EK4RRP.npy', 'L66E2921S3O1MURX.npy',
                 'BCPN2Y1Q3287F27T.npy', 'OXRK4I6CTXCENAZV.npy',
                 'TRRYZ5WXYHUMTPCQ.npy', 'JLWQD6J6FXTRVZ0N.npy',
                 'SZY3J4UTKSCFM2H4.npy', 'NJZN2TNXUUFRZ3GZ.npy',
                 '2WN4Y1MAS89PSM9P.npy', 'FK3914JSRYGJ80AO.npy',
                 'ZHLGEIY4UAEIPL33.npy', 'FA9ITR81JQB372QO.npy',
                 'HIAEQWMTUGATOAVI.npy', 'RL7MPCDASC5FXBRJ.npy',
                 'THOT71FJ6S8OS792.npy', 'KHPDR2WKRBY6ETGF.npy',
                 'IAPTWF22G2YMU1P6.npy', 'SUOSQFF89IZTMJTG.npy',
                 'QGH1NCRAPQ9QI8DG.npy', 'METEKE1H72A6VT8I.npy',
                 'JWV4HKJ6OCKS26IR.npy', 'MEP2WAB3JNIXJTON.npy',
                 '5IU465J6DS71MDL0.npy', 'CCGEZOFGLKC3USJ8.npy',
                 'ROF7JT3SZGBNY3MG.npy', 'ZVRO2HKWM7JSS6IA.npy',
                 'ZUR2IKTUC2KQVB2X.npy', 'ALOUY4SF3BQKXQCZ.npy',
                 'PMUAC5NYSRLNM8NZ.npy', 'WDIYHNMBALZ4ZP2A.npy',
                 'AVK7I5XLSCRWM6FN.npy', 'OTIGVYUMMBXDNSHN.npy',
                 'KRH0TOXLZZFQHS3N.npy', '82KAG7KXLABZCMKJ.npy',
                 'SWSY42N45J3YWV5I.npy', 'Q9K5IEJR20DGZMSZ.npy',
                 'DL9L2SNS6U80DI1A.npy', 'BTLLL3I67PYRN7R4.npy',
                 'L37U6THCBS65YAA9.npy', 'ZAJH6DQQBAECJ7AX.npy',
                 'NRD1J2CBBXXGPNFL.npy', 'HAKJSG5CHDGOBP6B.npy',
                 'SBRRR2SATWOOR5D6.npy', 'LRCPYDPZFFC6HR6K.npy',
                 'ENB4V4BPR4HRCRXW.npy', 'LYUO2OPTNYUBCHT2.npy',
                 'RHOULE6IWB6KWLE5.npy', 'RLF1AFE0G1Q98BS9.npy',
                 'C5ZH8SM0890PU4DL.npy', 'VVGO45TQNOASBLZM.npy',
                 '7A455JLNBCTEBIYS.npy', 'UAFTPLA47GA7WS27.npy',
                 'ADJYK6PUQSLE2GLN.npy', '6I1FUOBB7KQIA5L2.npy',
                 'DNM7TGJUNGRPMZZE.npy', 'OMR7KPUBB2T7X3QQ.npy',
                 'IXMOCOA4Z88DGJ6E.npy', 'GHNKISIFB2PFOZ8O.npy',
                 'BI8RLR13ROJPHFM0.npy', 'QTUPNSXYST53IETT.npy',
                 'TRM3ANQXS7O6Y7OA.npy', 'SEV2GAEFNJBLX3ZJ.npy',
                 'HGJTE6ROVCUFUC4S.npy', 'WBU2NUH7X0X5J4RN.npy',
                 'CQARGKFSIVQWNCT6.npy', 'D6PTKXBC1QKIPLDM.npy',
                 'K5584RWAW774MYAM.npy', 'BSKAYXMTSEXR6U5K.npy',
                 'SZKV26SJOHXJBPVX.npy', 'HBAZOCWOGBJYCEAZ.npy',
                 '2U95HNIBIUWTDGF6.npy', 'FYXMPCGHTGJXJBSD.npy',
                 'J3JHPC3E15NZGX34.npy', 'NGQAJ2GBHZE2DN5C.npy',
                 'XGUULATQ6C4ZP3QQ.npy', 'MJFHYCVEV2GTJD3N.npy',
                 'ADKFJYDHLE7M3XE6.npy', 'HVPI7QQSXPDHSFME.npy',
                 '3LR0UAB5II6XJ0W8.npy', '5KZSOKNYS84ZTDK6.npy',
                 '0OJARUOGIFH5TZKU.npy', 'AEPRN5R7W2ASOGR0.npy',
                 'KBY6B17YK6AAR3VD.npy', '5YO8QRZ4B3RE047F.npy',
                 'TUI0SHMO3IO4PT8Z.npy', 'QGEQ4IFSRIA3RBVB.npy',
                 'JQVBM4FUT34IUSUB.npy', 'ZTFA5BJQHNJIRH4T.npy',
                 'BRFHLOM2KPUXSBMI.npy', 'OI623KTR79Z90BG1.npy',
                 '8XJTSY1KAY150OIS.npy', 'YSU22E0MX5CRK3XC.npy',
                 'EQ0L201MUZFO2SY8.npy', 'NWO33W453F6ZJ4BU.npy',
                 'PMUXVQHYYYBOK4RL.npy', 'DZOMCPYXT1TY592W.npy',
                 'HAD27IUS6FJC8BCO.npy', 'POYYSD3RSJFBKSLQ.npy',
                 'SODFDRR2LWR7JDMN.npy', 'GJ35FZQ5DSP09A4L.npy',
                 'GET5PY3CH6RJV3UW.npy', 'OLCBKEY2WKCWPHJW.npy',
                 'HCA8NOIQCJL1EQLK.npy', 'YGJSUAFCVJ52LIRD.npy',
                 '5ONG9EN0RF9CYGFY.npy', 'BU830YLY7PZLQZIT.npy',
                 'OCW3VKYNF3YM4CCK.npy', 'HFESZX9P9R6TLSSW.npy',
                 '6SFFLWAD7TCCTTHC.npy', 'TFLSOP3FLUBQDLL2.npy',
                 'QYUTY0NUYHCFCWY6.npy', 'GHD3U9CYRYQ4A0A0.npy',
                 '8LL3IE6PJ9GC2JX0.npy', 'BLPHSKRODCJLHZZS.npy']
VALIDATION_LIST = ['IWYDKUPY2NSYJGLF.npy', 'YBMFJQLVZENVF6MA.npy',
                   '6FQBJ7LCRC0C67HG.npy', 'QA48KZ7WBFX9ZOIA.npy',
                   'K2GS9PIQ1E0DBDBE.npy', 'FPTFJD4JZA7ZKYQJ.npy',
                   'TNUEVMD3Z4YZFQDG.npy', 'ZCMTUATV42LAXACO.npy',
                   '1RL0HZOC3A7JWJL1.npy', 'LHMGXWC6J7DRD3CP.npy',
                   'TLNUR0GUPFTO4Y73.npy', 'R9RD6OJJBJCI7L25.npy',
                   'ZZWCWIY01XXP36WL.npy', 'EGCV7CJ6LF2CACQM.npy',
                   'C02TH0J1FT7XQ5EU.npy', 'IOK4CHCPSG7V7V5Y.npy',
                   'NFG67H47EUPD5URG.npy', '45BKZ74PBPRK5Q46.npy',
                   'WCSSH2NHDNNZIVML.npy', 'SVYZISKLZLQKBPX6.npy',
                   'HLQKXLS4OQ2BGQ5C.npy', 'KNKVPM2UV4UFFC5R.npy',
                   'XPJH44FSBRVL2WTV.npy', 'XZUUUC426GUXSCGJ.npy',
                   '2KMKXR2G1BLD0C2G.npy', 'XQLUVUDY23UZMYW3.npy',
                   'LAUIHISOEZIM5ILF.npy', 'SUVCYHBWWKNAVUGQ.npy',
                   'KMDW8QCA7WO2NWTI.npy', 'OCD6VASBJF4SNZ4U.npy',
                   'WZDKIWF6GHR3J2GJ.npy', 'ASI5JI7DJLB3RKFS.npy',
                   '54IOGT037BUFO9QN.npy', '0YSFCN6QTA52Q013.npy',
                   'TITRFCZEPFJKM3NY.npy', 'AAME0ZF2CI8QISTT.npy',
                   'JUM3FRXBYLTATYYH.npy', '9WDSZR9939UAVNUL.npy',
                   'CSXRYWJCXJXAJRLP.npy', 'XUGDI7AE3XNZ45SU.npy',
                   'MPAR5LKXIUS2RTWG.npy', 'FFVM6VCPRIKK7453.npy',
                   'GB9QW6WTB9XYMMN4.npy', 'VJNCTEEYR7U4HIQU.npy',
                   '4R0C9IQ512KYCGPC.npy', 'HHM4X4CF7WXJXSCT.npy',
                   'CRH6GLC3CTD2WACN.npy', 'GONORRCZWHDZV3M5.npy',
                   'TQCRCH9LFKMCFRUY.npy', 'EK0MO7OMQQEW0UYA.npy',
                   'XSSFSN7XYAV4E3OA.npy', 'DHG4RIY13DFISJ35.npy',
                   'WPNYHBQRGJ4NHU3C.npy', 'FTAKNRFGOGVS632K.npy',
                   '1DGYAK7UR0OR070J.npy', 'UUKSYXBYQQZATBP5.npy',
                   '2WEAS2OYMHM0QP0M.npy', '7E6EK7VVSNAESVYS.npy',
                   'DENODP22U2Q7E4LM.npy', 'BCYM2RGUQ2W0HKEH.npy',
                   'VDO7JNWOEJ5KGJYJ.npy', 'AAU7FDO320X6D4A2.npy',
                   '5H94IH9XGI83T610.npy', 'VSJRIEOFJCVJIO2C.npy',
                   'NJPOTXDCGONCOU7C.npy', '5JR74HUQF0GU82S9.npy',
                   'JJEB6MFMSB6UAKHU.npy', 'PMEECXFHBLN6KHO7.npy',
                   'LVOHLK3FOHWE66BP.npy', 'UOZJSIPCXRA7UJF2.npy',
                   'PAHYRPTLLQOU2FTB.npy', 'KK2Y9XHUUUC5LISA.npy',
                   'UTQXBYIUWME8LK2K.npy', 'YMTB3QZYNO0L8JB2.npy',
                   'KSPGRITHL4N0XPLW.npy', 'NMHOGEKTK7LT57ZI.npy',
                   'JT7TUT2NRR0DPZZF.npy', 'IAX2O3NWENSSH4D5.npy',
                   'GQ6UNFKEJD1D4V7X.npy', 'NLV36DRLRZY5AKIJ.npy',
                   '5FY8Y34MBL26F0EE.npy', '3L337CU2FCNMNBXH.npy',
                   'HB04152LQFDZPEBP.npy', 'XYOD23KJFH7NZXXK.npy',
                   'SUOA7XPFTTR2O5NQ.npy', 'ZCTE57VMPCGWDG75.npy',
                   'CWAL6Z1OAGM4URNC.npy', '2D6I54KTG2GY3FUW.npy',
                   'DXOU8FRR7I9BLBN9.npy', 'IIAFOQ3HGHU4A6UT.npy',
                   '9G0Y9ORHYQF87HGL.npy', 'QIA74LHB1XWMP523.npy',
                   'HAYSFX75ACG77ZJE.npy', 'SGFWUNFEYCKBMF65.npy',
                   'SE3O4E4F4Z70C75B.npy', 'NYNZ6UEFTJP4UNWP.npy',
                   'JNLCZIQGTGTFNEKK.npy', 'C67K82FMKN435ZNB.npy',
                   'Y4GJVIUVR4NAMLXW.npy', 'FIBU2GYR72UU7E5M.npy',
                   'YY183POZ782MZSZH.npy', 'BWDRAF5ZOQPFOOL7.npy',
                   '72IYRWNFFY7OFPJ7.npy', 'MYTHDP7XSELNJUBQ.npy',
                   'ZNPFNNZ2DRELKNQ3.npy', 'NHCZX3GOEMPDDHW3.npy',
                   '3LEMIRZE3FJWTWOT.npy', 'ZQYME7XWFWQXXOXA.npy',
                   'GBXENHYLRVZ5DTHI.npy', 'WHNAL8GW8TJ8Y4AK.npy',
                   'XHX3PCDOWEWDB34X.npy', 'OJC5WXUS4IEGGOYC.npy',
                   'LZU7NSTDRFX77U7G.npy', 'RKIZJRFZGCXD4EOL.npy',
                   'DEWCDZZM1CSH5YFI.npy', 'PXUXRMI8Q4TOF3FK.npy',
                   'SAQ39AN6XOILQY1D.npy', 'ICWMS7OCXZLUTUFT.npy',
                   'PCJ6DNFF7XHMUSWE.npy', 'DXLPP6QTNOHQRHVG.npy',
                   '3ZBJ0AG0IB9BZ6G5.npy', 'RW13S0OR03CO7OP5.npy',
                   'MA4Z8BDBS9XC75A5.npy', 'QLYPHVVU3RG43OZO.npy',
                   'UN6XC6HYXO0RDKU6.npy', 'I6QW7NPU65SKPIMZ.npy',
                   'UEIDAD48BPY82DY3.npy', 'RYNCGNVRZH2LB7BN.npy',
                   '580WPPT613GFF945.npy', 'NZ02ITT0KYQGCHKX.npy',
                   'ARZ7H7BSITHOSSMG.npy', 'UYA5YJUAVC3SE6HV.npy',
                   'BSBXPIGXEUV3WWOZ.npy', '4HJSEGIZANYH5ISF.npy',
                   'MZAZFTEIU4NLYXDF.npy', 'SKBIT57CPAKAGQVB.npy',
                   '2BK5XXUHXSEDPUTG.npy', '89U2QG6ISBDM51GI.npy',
                   'ADIO4U6QJTXF3YQH.npy', 'JD6C377DW9DEXDDY.npy',
                   'Q8LO6RYLTPQQKJO8.npy', 'DWWMOMZWT1DCBAVD.npy',
                   '1CZ6MRB4HLDZSWBO.npy', 'U82WU8SEKUCXCHZI.npy',
                   'UI0XEE9RCAEFSYP2.npy', 'YNLMIJ5S24AYYZXN.npy',
                   'SUENUHMZ5JN7IWJE.npy', '0EZJW4R4EMC16I10.npy',
                   '37ZPX63RET9P2QHM.npy', 'RTC5RXQHUXCSPNLN.npy',
                   '6GA622A55T3AOGTL.npy', '2ZF1T5U8LU7SMI4Y.npy',
                   '0LT6HMDHUK63WSC7.npy', 'FQJ4RD45AQ53MYRC.npy',
                   'YSTBBUD35YZBOT2Z.npy', 'DZTDN47H2IKSYN3Q.npy',
                   'YEXJFFF2ZK44FYCP.npy', 'ZXJH70C8PBEPE76K.npy',
                   'NCI33JITLERBTWXI.npy', 'PVMXD3RZSHEQTYW3.npy',
                   'OAC7JXARELEVVEQW.npy', 'VVVMGD8KR33MD3SA.npy',
                   'QBOBDYYD4R7WP37S.npy', '2STQ3IYP5SV7XIPG.npy',
                   'THKULF2KHMNVMTUR.npy', 'OUM3NFGU5U3TCBBK.npy',
                   'XJJ3IXEOPNFXBGP7.npy', 'ESLHGQCGYB7J4DX3.npy',
                   'PHMZRI2MBR54TE7A.npy', 'KACA7XHWXRM6A3W3.npy',
                   'M5TTZY7064N29GGE.npy', 'ZX3B62KA1WW46Z6O.npy',
                   'KD1MCH6J7162T5QA.npy', '7CX50JMLP3MB8Y6X.npy',
                   'EXOIBZU8JEIWNCHS.npy', 'T4C94OKPNJ1QJHEI.npy',
                   'JJMENP4QE4CSXSHV.npy', 'BVYSR54MKT5SPMSU.npy',
                   'JWKB7SHIBYWSEVMC.npy', 'IIOOPK6T2ZWDZMU3.npy',
                   'NVKJ2CHCS3P7QLEK.npy', 'GBA7XJ5E7ITWZW7K.npy',
                   'RMID4XEKTVEYCSGK.npy', '7O84UHF46Z2E5AJ0.npy',
                   '8LWUQKCNESCFMX7J.npy', 'ZZX0ZNWG6Q9I18GK.npy',
                   'FMYU2WQIQCFQBIFS.npy', 'CJDXGZHXAGH7QL3C.npy',
                   'GFFSQZ18XGOVEET0.npy', 'OSO6K4FGNA57I3VJ.npy',
                   'NYR7NWGHIRZDCL66.npy', 'KGK66IWDH2EX5XNP.npy',
                   '639OS2YC4LW8LFDJ.npy', 'X4GQ56MK0E0YNL0E.npy',
                   'WB6U44ZD7QACX6JM.npy', 'SP73CHWDY57ND02Z.npy',
                   'X63IP208CJ8ST0SX.npy', 'SSJQF5OU3VSC3HAD.npy',
                   '3BCIGI0WUQN3QIOZ.npy', 'AJHYOJIG3DZLSIKI.npy',
                   'GOXKSLNERTQ7DSPQ.npy', 'NUWPDW6HBRTLEYWD.npy',
                   'ZAHC7FTFP8GK5JBE.npy', 'I0CYK5NWY1M896AK.npy',
                   'J2JPFK8ZOICHFG34.npy', 'RBC423N4RNB3UZ5P.npy',
                   '647D5SO6MJEHAT40.npy', 'QPDX2K3DS7IS5QNM.npy',
                   'XPBRBUB6YYHIWHVO.npy', '8ZK7DFZXQM8IK5BL.npy',
                   'ANZM4SIQFTWG7K47.npy', 'HZLBHRHYLSY9TXJ4.npy',
                   'RFX3MOOO42O26FFD.npy', 'G37FBCOEGUZFY138.npy',
                   'MGHOPQDA7YCI7643.npy', 'Z8KUSR13R8Z4Y5KH.npy',
                   'BNZ2UKOA43BESZGP.npy', '0DQO9A6UXUQHR8RA.npy',
                   'JQWIIAADGKE2YMJS.npy', 'ILO1WFJV843ZHAME.npy',
                   '8C0V0CZ2BRLSAMUM.npy', 'HYAJCS9Z3PFLOTET.npy',
                   'RXK7K2FPVPLFMID2.npy', 'E34MR3LIUGH22Y7D.npy',
                   'TXP81XXA11Z8BNIX.npy', 'LOZOKQFJMDLL6IL5.npy',
                   'TNXMNBGRZHPIMJJI.npy', 'A096PH8X9XCTJECG.npy',
                   'GCSZPU5LPJII8FGU.npy', 'KFX44Z5T5RWT4LLI.npy',
                   'KQF9NGGWELW9GCUP.npy', '4EH0PS1IKFIGT1T0.npy',
                   'PAIDKQGSUI0KZ54S.npy', 'I0HKAX9TAXJAO1OW.npy',
                   'SLI467X57QG8JLXX.npy', 'SLPTPLSMZYJEUEXK.npy',
                   'ZANU4GPPZNIFRN6W.npy', 'CJNLJB43ZD3GIUN3.npy',
                   'PB7YJZRJU74HFKTS.npy', '9BKRXCC8B2A9TFRA.npy',
                   'BMLCCAQRWKFKZQTC.npy', '8HMXW6GR38KFRF2T.npy',
                   'MEQKZCRF5TR6MHLY.npy', '33A2LR9XK8O2FO0S.npy',
                   'HDQYSVBJ4MPRGYRN.npy', 'LINKQMUO9DQ43BNH.npy',
                   'CGQFS5PB627OWLL7.npy', 'BHCMG7A3E5Y13T0J.npy',
                   '6Y2HYY248E0T6HLO.npy', 'UKOHM6O57N37U2E2.npy',
                   'FBGMN3O08GW5GG91.npy', 'WPA6WCTE24HGRIXH.npy',
                   'CJ678Q2IJCD4BSYJ.npy', 'E6KN06NUXQQIP32L.npy',
                   'RHWTMTHC7HIWZ2YZ.npy', '9UIAZ2U1711BN4IW.npy',
                   'RSKIY1U4X5QAUAAK.npy', 'EKKECBQEXHI5ME2T.npy',
                   '56GX2GI8AGT0BIHN.npy', 'ILNTKMBVTXNXURGV.npy',
                   'LIJBIVVBI6TJCT6Z.npy', 'TFXM6VTRLJH2DS7G.npy',
                   'JSV6UEFKJAVM6OGO.npy', 'Z1GHH2QNG3BGMP4J.npy',
                   '37M2EWHIRFGWOP33.npy', 'JP2IRK8910EXF0IW.npy',
                   'PI11PYBO46QC196J.npy', '6Y3CL8LU6OXAQEHL.npy',
                   'RODVAIJ6A6P4D54Y.npy', 'EUNTRXNEDB7VDVIS.npy',
                   'ZMEGS4G44P6FMONO.npy', 'MPTG3X65O7O7HYHX.npy',
                   'DNO6CS3YNGFMUWXL.npy', 'HXBUUFVQBNISYKNU.npy',
                   '26UJHQ45YJOBI1WA.npy', '9FEMEHY0NPT0UW92.npy',
                   'AZKCIPFZYXES2I5R.npy', 'NHKUCEIIHLJQ25VM.npy',
                   'JCWGEJIJO2LDRY2U.npy', 'RCDAHVLV6QUTDFVD.npy',
                   'YXCAWWW2FRTZ3C38.npy', 'AGTRMFUZM2MQUAB4.npy',
                   'LGFNFIWO2ZEQYK36.npy', 'KPC2RVJBDHQLC5EM.npy',
                   '8KZMN84B4R5RPP2Q.npy', 'XWC4D3BST0D6WMOT.npy',
                   'BHA4CDH32NPSOKTZ.npy', 'WHDZVHCOCOTIJLXV.npy',
                   'MG26HSJJ5R1ACK45.npy', 'BMTJZEMYGJCVHX7N.npy',
                   'RJIM7Y5H8XML9CA7.npy', 'SZXTZ3LQ5MBFDFXR.npy',
                   '3BJGJJBGH5TRF8PP.npy', 'YYNFU45K2WIMXWE4.npy',
                   'F8W2RDY3D6L2EOFT.npy', 'RATVCRHG4ZHLCOV3.npy',
                   'POSX6PS6LR34HENB.npy', 'SADEJMD27KKWZQ5L.npy',
                   'PIKNNHQAI52EI62K.npy', 'DMQR502XO4Z5DSM9.npy',
                   'ZN47YAP6WCJ3M4OB.npy', 'RWJZNVFII5INU6Z5.npy',
                   '48XO6UQZI766F8TN.npy', '9DWQA9BENM1XBI71.npy',
                   'ZCIMXZ73NFNRDH46.npy', 'VFHSASVT4EXZWAOS.npy',
                   'IZF32FDSYNLK72D1.npy', 'KOQ6SDQIBEZAOCFQ.npy',
                   '6UI52CFMWHR6UH22.npy', 'IKLKZYXDYTCCYUAZ.npy',
                   'IC5A7GCAADV0CB0Q.npy', 'EWYGH3OSPLJI27UF.npy',
                   'UOSAPEMY76KAGOUR.npy', 'G7ZGNCQ8PR5NXN9Z.npy',
                   'GJ35BUS1EQT27H5F.npy', 'ZUMTGMC37Y4THDDT.npy',
                   '6QXS535HGCT3QU7Z.npy', 'LOGIP43ZSLSUA5SZ.npy',
                   'VUTDS6OC2VUSNQMK.npy', 'XJLRFX244JGJZUUF.npy',
                   'COFP4UFH6MWZA7HI.npy', 'PQJEIXKIJMZTBZ74.npy',
                   '6IPDMHP9V2TQJ9YR.npy', 'ZUEK5YSS7CITVWIP.npy',
                   'DDPQNEBC34XCOF2U.npy', 'Q8BNE59JIKQLLYJ1.npy',
                   'QNAGLY5L1PQ4Y2QS.npy', 'IXBFNU6QB2SEZQ7N.npy',
                   'ZIDRKXRWTVGHW7ZD.npy', 'WL52KFPE6FMJ1A1I.npy',
                   'PBR2STY62FZ3MQHZ.npy', 'ATY6WNE6JVYJ2RLN.npy',
                   'GQNS07SL34K9IUIU.npy', 'DXOUKNW7AXCZFUIL.npy',
                   'H4C8M2RF0ASI8M5R.npy', 'PYZP7WQ4QRH46EFS.npy',
                   'MOAADIHO5RHQJEK3.npy', 'R1556ANHLYF14MO8.npy',
                   'AHNGBFJSTEBO4BJW.npy', '6RPWFJ3A9FXOZC7E.npy',
                   'L7PQTICOQFQXQQH7.npy', 'R0GUIL1D7Q8UULH2.npy',
                   'ECZZ9D6T0D9K48X6.npy', '76KZMJS23TOIPULM.npy',
                   '0QPSB9IO98216B98.npy', 'EURBSIKPYU66GBRW.npy',
                   'ENPUI7ZVHUGIXYN7.npy', 'SSQSB70QYC5L5CJJ.npy',
                   'AGTDPQEU54ODFAUD.npy', 'RZVMIO4YX2JHAUQ5.npy',
                   'I5LGG0BM4IP16DPT.npy', 'Y4LKD502EZ8ICYLI.npy',
                   'Z3AINLH4Y07ITBRR.npy', 'BJI23MUAPMUMIED2.npy',
                   'HW93K0TDEQO9XO88.npy', 'OTIP25YORHBKJPZM.npy',
                   'IYDXJTFVWJEX36DO.npy', 'TSDXCC6X3M7PG91E.npy',
                   '3YQNRP7QKLAD6YN5.npy', 'TJAPUMPEDA32J5JW.npy',
                   'NQXVKFP54XTD3GVF.npy', 'UNREF6O6622LZWNN.npy',
                   'U5ZUK9GYCAM4FYK4.npy', 'NHXCOHZ4HH53NLQ6.npy',
                   'MTWW42SPCGLHEDKY.npy', '0RB9KGMO90G1YQZD.npy',
                   'QTMN6M3FY3YSJ6ZP.npy', '24D888C8JHJ4RAI9.npy',
                   'KSCM1YWI4TPRYRM3.npy', 'Q0YMNAR31STBFPG6.npy',
                   'RAHQRN78MNIOX7RU.npy', 'UCORZR684EMJSR95.npy',
                   'LK60A6W87BBO0GSB.npy', 'KK0ZHCZR9V87OESW.npy',
                   'IAUKV5R644JZFD55.npy', 'RI48TVZ6U35LMAB8.npy',
                   'CCVIH3DIM3GF6SLG.npy', 'QJFMNVKKVZXKBROO.npy',
                   'BNAGD36PDWFHIEOO.npy', 'RWKA32WBSVFB4MQF.npy']


def mip_array(array: np.ndarray, type: str) -> np.ndarray:
    # for numpy data:
    # return np.max(array, axis=0)

    # for preprocess_luke data:
    return np.max(array, axis=2)


def crop(arr: np.ndarray):
    to_return = arr[151:len(arr)-51]
    return to_return


def remove_extremes(arr: np.ndarray):
    a = arr > 270
    b = arr < 0
    arr[a] = -50
    arr[b] = -50
    return arr


def download_array(blob: storage.Blob) -> np.ndarray:
    in_stream = io.BytesIO()
    blob.download_to_file(in_stream)
    in_stream.seek(0)  # Read from the start of the file-like object
    return np.load(in_stream)


def normalize(image, lower_bound=None, upper_bound=None):
    if lower_bound is None:
        lower_bound = image.min()
    if upper_bound is None:
        upper_bound = image.max()

    image[image > upper_bound] = upper_bound
    image[image < lower_bound] = lower_bound

    return (image - image.mean()) / image.std()


def upload_png(arr: np.ndarray, id: str, type: str, bucket: storage.Bucket):
    """Uploads MIP PNGs to gs://elvos/mip_data/<patient_id>/<scan_type>_mip.png.
    """
    for i in range(len(arr)):
        try:
            out_stream = io.BytesIO()
            misc.imsave(out_stream, arr[i], format='png')
            out_filename = f'mip_data/{id}/{type}_mip.png'
            out_blob = storage.Blob(out_filename, bucket)
            out_stream.seek(0)
            out_blob.upload_from_file(out_stream)
        except Exception as e:
            logging.error(f'for patient ID: {id} {e}')


def save_npy_to_cloud(arr: np.ndarray, id: str, type: str):
    """Uploads MIP .npy files to gs://elvos/mip_data/<patient_id>/<scan_type>_mip.npy
    """
    try:
        print(f"gs://elvos/mip_data/{id}/{type}_mip.npy")
        np.save(file_io.FileIO(f'gs://elvos/mip_data/{id}/{type}_mip.npy', 'w'), arr)
    except Exception as e:
        logging.error(f'for patient ID: {id} {e}')


if __name__ == '__main__':
    configure_logger()
    client = storage.Client(project='elvo-198322')
    bucket = storage.Bucket(client, name='elvos')

    in_blob: storage.Blob
    # .npy
    # for in_blob in bucket.list_blobs(prefix='numpy/'):
    # luke
    for in_blob in bucket.list_blobs(prefix='preprocess_luke/training/'):

        logging.info(f'downloading {in_blob.name}')
        input_arr = download_array(in_blob)
        logging.info(f"blob shape: {input_arr.shape}")

        # .npy
        # cropped_arr = crop(input_arr)
        # not_extreme_arr = remove_extremes(cropped_arr)
        # luke
        not_extreme_arr = remove_extremes(input_arr)

        logging.info(f'removed array extremes')
        # create folder w patient ID
        axial = mip_array(not_extreme_arr, 'axial')
        logging.info(f'mip-ed CTA image')
        normalized = normalize(axial, lower_bound=-400)
        plt.figure(figsize=(6, 6))
        plt.imshow(axial, interpolation='none')
        plt.show()

        save_npy_to_cloud(axial, in_blob.name[25:41], 'axial')
        logging.info(f'saved .npy file to cloud')

        # sagittal = mip_array(not_extreme_blob, 'sagittal')
        # upload_png(sagittal, in_blob.name[6:22], 'sagittal', bucket)
        # save_to_cloud(sagittal, in_blob.name[6:22], 'sagittal')
        #
        # # save
        # coronal = mip_array(not_extreme_blob, 'coronal')
        # upload_png(coronal, in_blob.name[6:22], 'coronal', bucket)
        # save_to_cloud(coronal, in_blob.name[6:22], 'coronal')

        # save
