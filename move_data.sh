H=~/data/2026_09_22_Hessel
FA="$H/ForceData/RigA/2026_0921_Kiisa_Complete/Data"
FB="$H/ForceData/RigB/2026_0921_Kiisa_NotComplete/Data"
T="$H/EDL_Lo"

# m <xray dir relative to $H> [force file path without extension]
m() {
  local name=$(basename "$1") sub
  case "$name" in *[Aa]ctiv*) sub=Active;; *[Pp]assive*) sub=Passive;; *) echo "?? $name"; return 1;; esac
  mv -nT "$H/$1" "$T/$sub/$name" || return 1
  [ -n "$2" ] && cp -nv "$2".d?? "$T/$sub/$name/"
}

# --- Day1 WTMouse2 ---
m Day1/WTMouse2/RigA_LoActiveAgain1_0022   "$FA/WT2 RigA/Lo_PassiveAgain1"
m Day1/WTMouse2/RigB_LoPassive1_0025       "$FB/WT2/W2_RigB_LoPassive1"
m Day1/WTMouse2/RigB_LoActive1_0026
m Day1/WTMouse2/RigB_LoAgainPassive1_0029
m Day1/WTMouse2/RigB_LoAgainActive1_0030   "$FB/WT2/W2_RigB_LoActiveAgain1"

# --- Day1 WTMouse3 ---
m Day1/WTMouse3/WT3_RigB_LoPassive1_0052   "$FB/WT3/WT3_Lo_Passive1_Xray"
m Day1/WTMouse3/WT3_RigB_LoPassive2_0053   "$FB/WT3/WT3_Lo_Passive2"
m Day1/WTMouse3/WT3_RigB_LoPassive3_0054   "$FB/WT3/WT3_Lo_Passive3"
m Day1/WTMouse3/WT3_RigB_LoActive1_0055    "$FB/WT3/WT3_Lo_Active1"
m Day1/WTMouse3/WT3_RigB_LoActive2_0056    "$FB/WT3/WT3_Lo_Active2"

# --- Day1 WTMouse4 ---
m Day1/WTMouse4/WT4_RigA_LoPassive1_0070   "$FA/WT4/LoPassive1"
m Day1/WTMouse4/WT4_RigA_LoPassive2_0071   "$FA/WT4/LoPassive2"
m Day1/WTMouse4/WT4_RigA_LoActive1_0072    "$FA/WT4/LoActive1"
m Day1/WTMouse4/WT4_RigA_LoActive2_0073    "$FA/WT4/LoActive2"

# --- Day2 WT5 ---
m Day2/WT5/WT5_RigA_LoPassive1_0193        "$FA/WT5/RigA/LoPassive1"
m Day2/WT5/WT5_RigA_LoActive1_0194         "$FA/WT5/RigA/LoActivr1"
m Day2/WT5/WT5_RigA_LoActive2_0195         "$FA/WT5/RigA/LoActivr2"
m Day2/WT5/WT5_RigB_LoPassive1_0208        "$FB/WT5_RigB/LoPassive1"
m Day2/WT5/WT5_RigB_LoPassive2_0209        "$FB/WT5_RigB/LoPassive2"
m Day2/WT5/WT5_RigB_LoActive1_0210         "$FB/WT5_RigB/LoActive1"
m Day2/WT5/WT5_RigB_LoActive2_0211         "$FB/WT5_RigB/LoActive2"

# --- Day2 WT6 ---
m Day2/WT6/WT6_RigA_LoPassive1_0223        "$FA/WT6_RigA/LoPassive1"
m Day2/WT6/WT6_RigA_LoPassive2_0224        "$FA/WT6_RigA/LoPassive2"
m Day2/WT6/WT6_RigA_LoActive1_0225         "$FA/WT6_RigA/LoActive1"
m Day2/WT6/WT6_RigA_LoActive2_0226         "$FA/WT6_RigA/LoActive2"
m Day2/WT6/WT6_RigB_LoPassive1_0237        "$FB/WT6_RigB/LoPassive1"
m Day2/WT6/WT6_RigB_LoPassive2_0238        "$FB/WT6_RigB/LoPassive2"
m Day2/WT6/WT6_RigB_LoActive1_0239         "$FB/WT6_RigB/LoActive1"

# --- Day2 WT7 folder (WT7 + WT8) ---
m Day2/WT7/W7_EDL_RigA_LoPassive1_0264     "$FA/WT7/EDL/RigAEDL_LoPassive1"
m Day2/WT7/W7_EDL_RigA_LoActive1_0265      "$FA/WT7/EDL/RigAEDL_LoActive1"
m Day2/WT7/W7_EDL_RigA_LoActive2_0266      "$FA/WT7/EDL/RigAEDL_LoActive2"
m Day2/WT7/W8_RigA_EDL_LoPassive1_0295
m Day2/WT7/W8_RigA_EDL_LoPassive2_0296
m Day2/WT7/W8_RigA_EDL_LoPassive3_0297     "$FA/WT8_RigA/Lo_Passive1"
m Day2/WT7/W8_RigA_EDL_LoActive1_0298      "$FA/WT8_RigA/Lo_Active1"
m Day2/WT7/W8_RigA_EDL_LoActive2_0299      "$FA/WT8_RigA/Lo_Active2"
m Day2/WT7/WT7_RigB_LoPassive1_0011        "$FB/WT8_RigB/LoPassive1"
m Day2/WT7/WT8_RigB_LoPassive2_0012        "$FB/WT8_RigB/LoPassive2"
m Day2/WT7/WT8_RigB_LoActivate1_0013       "$FB/WT8_RigB/LoActive1"

# --- Day3 WT9 ---
m Day3/WT9/RigA_WT9EDL_LoPassive1_0039     "$FA/WT9_RigA_EDL/LoPassive1"
m Day3/WT9/RigA_WT9EDL_LoActive1_0040      "$FA/WT9_RigA_EDL/LoActive1"
m Day3/WT9/RigA_WT9EDL_LoActive2_0041      "$FA/WT9_RigA_EDL/LoActive2"
m Day3/WT9/RigB_WT9EDL_LoPassive1_0044     "$FB/WT9_RigB/EDL_LoPassive1"
m Day3/WT9/RigB_WT9EDL_LoPassive2_0045     "$FB/WT9_RigB/EDL_LoPassive2"
m Day3/WT9/RigB_WT9EDL_LoActive_0046       "$FB/WT9_RigB/EDL_LoActive1"
m Day3/WT9/RigB_WT9EDL_LoActive2_0047      "$FB/WT9_RigB/EDL_LoActive2"

# --- Day3 WT10 ---
m Day3/WT10/WT10_RigA_Lopassive1_0057      "$FA/WT10RigA/LoPassive1"
m Day3/WT10/WT10_RigA_Lopassive2_0058      "$FA/WT10RigA/LoPassive2"
m Day3/WT10/WT10_RigA_LoActive1_0059       "$FA/WT10RigA/LoActive1"
m Day3/WT10/WT10_RigB_LoPassive1_0068      "$FB/WT10_RigB/LoPassive1"
m Day3/WT10/WT10_RigB_LoPassive2_0069      "$FB/WT10_RigB/LoPassive2"
m Day3/WT10/WT10_RigB_LoActive1_0070       "$FB/WT10_RigB/LoActive1"
m Day3/WT10/WT10_RigB_LoActive2_0071       "$FB/WT10_RigB/LoActive2"

# --- Day4 WT11 Rig A (first prep) ---
m Day4/WT11EDL11ALoPassive1_0197           "$FA/WT11_RigA/WTEDL11ALoPassive1"
m Day4/WT11EDL11ALoActive11_0198           "$FA/WT11_RigA/WTEDL11ALoActive1"
m Day4/WT11EDL11ALoActive2_0199

# --- Day4 WT11 Rig A (second prep, A2) ---
m Day4/WTEDL11A2LoPassive1_0205
m Day4/WTEDL11A2LoPassive1_0206            "$FA/WT11_RigA2/WTEDL11A2LoPassive1"
m Day4/WTEDL11A2LoActive2_0207             "$FA/WT11_RigA2/WTEDL11A2LoActive2"

# --- Day4 WT12 Rig B ---
m Day4/WTEDL12BLoPassive1_0219             "$FB/WT12_RigB/WTEDL12BLoPassive1"
m Day4/WTEDL12BLoActive1_0220              "$FB/WT12_RigB/WTEDL12BLoActive11"
m Day4/WTEDL12BLoActive2_0221              "$FB/WT12_RigB/WTEDL12BLoActive2"
m Day4/WTEDL12BLoActive3_0222              "$FB/WT12_RigB/WTEDL12BLoActive3"

# --- Day5 WT15 Rig A ---
m Day5/Day5_WT15_EDLALoPassive1_0428       "$FA/WT15EDL_Bleb/WT15_EDLALoPassiveBleb1"
m Day5/Day5_WT15_EDLALoPassive2_0429       "$FA/WT15EDL_Bleb/WT15_EDLALoPassiveBleb2"
m Day5/Day5_WT15_EDLALoPassive3_0430       "$FA/WT15EDL_Bleb/WT15_EDLALoPassiveBleb3"
m Day5/Day5_WT15_EDLALoActive1_0431        "$FA/WT15EDL_Bleb/WT15_EDLALoActiveBleb1"

# verify: should list only run folders (no nesting), and show which ones lack a .ddf
for s in Active Passive; do echo "== $s ($(ls $T/$s | wc -l))"; for d in $T/$s/*/; do printf '%-40s %s\n' "$(basename $d)" "$(ls $d | grep -E '\.ddf$' || echo '-- no ddf --')"; done; done
