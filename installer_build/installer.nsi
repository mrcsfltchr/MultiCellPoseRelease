!include "MUI2.nsh"
!include "LogicLib.nsh"
!include "nsDialogs.nsh"
!include "FileFunc.nsh"

Name "MultiCellPose"
OutFile "MultiCellPose_Installer.exe"
InstallDir "$LOCALAPPDATA\MultiCellPose"
RequestExecutionLevel user

Var Dialog
Var ServerAddressInput
Var ServerPortInput
Var ServerAddress
Var ServerPort
Var InstallMode
Var GpuModeRadio
Var CpuModeRadio
Var GpuInfoLabel
Var LogFile
Var CondaPath

!define MUI_ABORTWARNING
!define MUI_ICON "..\cellpose\logo\cellpose.ico"
!define MUI_UNICON "..\cellpose\logo\cellpose.ico"

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "..\LICENSE"
Page custom custom_page_creator leave_custom_page
Page custom gpu_page_creator leave_gpu_page
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

!insertmacro MUI_LANGUAGE "English"

Function pre_custom_page
  StrCpy $ServerAddress "127.0.0.1"
  StrCpy $ServerPort "50051"
FunctionEnd

Function custom_page_creator
  Call pre_custom_page
  !insertmacro MUI_HEADER_TEXT "Server Connection" "Enter the remote server details."
  nsDialogs::Create 1018
  Pop $Dialog
  ${NSD_CreateLabel} 0 5u 100% 12u "Remote inference server address:"
  Pop $0
  ${NSD_CreateText} 0 20u 70% 12u $ServerAddress
  Pop $ServerAddressInput
  ${NSD_CreateLabel} 0 40u 100% 12u "Server port:"
  Pop $0
  ${NSD_CreateText} 0 55u 30% 12u $ServerPort
  Pop $ServerPortInput
  nsDialogs::Show
FunctionEnd

Function leave_custom_page
  ${NSD_GetText} $ServerAddressInput $ServerAddress
  ${NSD_GetText} $ServerPortInput $ServerPort
FunctionEnd

Function pre_gpu_page
  StrCpy $InstallMode "auto"
FunctionEnd

Function gpu_page_creator
  Call pre_gpu_page
  !insertmacro MUI_HEADER_TEXT "Compute Backend" "Choose GPU or CPU installation."
  nsDialogs::Create 1018
  Pop $Dialog
  ${NSD_CreateLabel} 0 5u 100% 20u "Detecting GPU..."
  Pop $GpuInfoLabel
  ${NSD_CreateRadioButton} 0 30u 100% 12u "Install GPU (CUDA) build"
  Pop $GpuModeRadio
  ${NSD_CreateRadioButton} 0 45u 100% 12u "Install CPU-only build"
  Pop $CpuModeRadio

  nsExec::ExecToStack 'powershell -NoProfile -Command "(Get-CimInstance Win32_VideoController | Where-Object { $$_\.Name -match \"NVIDIA\" } | Select-Object -First 1 -ExpandProperty Name)"'
  Pop $0
  Pop $1
  ${If} $0 == 0
    ${If} $1 != ""
      ${NSD_SetText} $GpuInfoLabel "Detected NVIDIA GPU: $1 (recommended: GPU)"
      ${NSD_SetState} $GpuModeRadio ${BST_CHECKED}
    ${Else}
      ${NSD_SetText} $GpuInfoLabel "No NVIDIA GPU detected (recommended: CPU-only)"
      ${NSD_SetState} $CpuModeRadio ${BST_CHECKED}
    ${EndIf}
  ${Else}
    ${NSD_SetText} $GpuInfoLabel "No NVIDIA GPU detected (recommended: CPU-only)"
    ${NSD_SetState} $CpuModeRadio ${BST_CHECKED}
  ${EndIf}
  nsDialogs::Show
FunctionEnd

Function leave_gpu_page
  ${NSD_GetState} $GpuModeRadio $0
  ${If} $0 == ${BST_CHECKED}
    StrCpy $InstallMode "gpu"
  ${Else}
    StrCpy $InstallMode "cpu"
  ${EndIf}
FunctionEnd

Section "Main" SecMain
  SetOutPath $INSTDIR

  File "setup_environment.ps1"
  File "requirements.txt"
  File /nonfatal "Miniconda-Installer.exe"
  File "..\setup.py"
  File "..\MANIFEST.in"
  File "..\README.md"
  File /r "..\cellpose"
  File /r "..\guv_app"
  File /r "..\cpgrpc"
  File /r "..\install"

  StrCpy $LogFile "$INSTDIR\install.log"
  FileOpen $9 "$LogFile" w
  FileWrite $9 "MultiCellPose installer log$\r$\n"

  DetailPrint "Detecting existing Conda..."
  FileWrite $9 "Detecting existing Conda...$\r$\n"
  StrCpy $CondaPath ""
  nsExec::ExecToStack 'powershell -NoProfile -Command "(Get-Command conda.exe -ErrorAction SilentlyContinue).Source"'
  Pop $0
  Pop $1
  ${If} $0 == 0
    ${If} $1 != ""
      StrCpy $CondaPath $1
      FileWrite $9 "Found Conda: $CondaPath$\r$\n"
    ${EndIf}
  ${EndIf}
  ${If} $CondaPath == ""
    ${If} ${FileExists} "$LOCALAPPDATA\Miniconda3\Scripts\conda.exe"
      StrCpy $CondaPath "$LOCALAPPDATA\Miniconda3\Scripts\conda.exe"
      FileWrite $9 "Found Conda: $CondaPath$\r$\n"
    ${ElseIf} ${FileExists} "$PROFILE\Miniconda3\Scripts\conda.exe"
      StrCpy $CondaPath "$PROFILE\Miniconda3\Scripts\conda.exe"
      FileWrite $9 "Found Conda: $CondaPath$\r$\n"
    ${ElseIf} ${FileExists} "$LOCALAPPDATA\anaconda3\Scripts\conda.exe"
      StrCpy $CondaPath "$LOCALAPPDATA\anaconda3\Scripts\conda.exe"
      FileWrite $9 "Found Conda: $CondaPath$\r$\n"
    ${ElseIf} ${FileExists} "$COMMONFILES\..\..\ProgramData\Anaconda3\Scripts\conda.exe"
      StrCpy $CondaPath "$COMMONFILES\..\..\ProgramData\Anaconda3\Scripts\conda.exe"
      FileWrite $9 "Found Conda: $CondaPath$\r$\n"
    ${ElseIf} ${FileExists} "C:\\Anaconda3\\Scripts\\conda.exe"
      StrCpy $CondaPath "C:\\Anaconda3\\Scripts\\conda.exe"
      FileWrite $9 "Found Conda: $CondaPath$\r$\n"
    ${EndIf}
  ${EndIf}

  DetailPrint "Preparing Miniconda..."
  FileWrite $9 "Preparing Miniconda...$\r$\n"
  ${If} $CondaPath == ""
    ${If} ${FileExists} "$INSTDIR\Miniconda-Installer.exe"
      StrCpy $0 "$INSTDIR\Miniconda-Installer.exe"
    ${Else}
      StrCpy $0 "$TEMP\Miniconda3-latest-Windows-x86_64.exe"
      nsExec::ExecToLog 'powershell -ExecutionPolicy Bypass -Command "Invoke-WebRequest -Uri https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -OutFile \"$$env:TEMP\\Miniconda3-latest-Windows-x86_64.exe\""' 
    ${EndIf}

    DetailPrint "Installing Miniconda..."
    FileWrite $9 "Installing Miniconda...$\r$\n"
    ExecWait '"$0" /S /D=$LOCALAPPDATA\Miniconda3'
    ${IfNot} ${FileExists} "$LOCALAPPDATA\Miniconda3\Scripts\conda.exe"
      FileWrite $9 "ERROR: Miniconda installation failed.$\r$\n"
      MessageBox MB_ICONSTOP "Miniconda installation failed. Please rerun the installer."
      FileClose $9
      Abort
    ${EndIf}
    StrCpy $CondaPath "$LOCALAPPDATA\Miniconda3\Scripts\conda.exe"
  ${EndIf}
  FileWrite $9 "Using Conda: $CondaPath$\r$\n"
  FileClose $9

  DetailPrint "Starting environment setup..."
  SetDetailsView show
  nsExec::ExecToLog 'powershell -ExecutionPolicy Bypass -File "$INSTDIR\setup_environment.ps1" -InstallDir "$INSTDIR" -ServerAddress "$ServerAddress" -ServerPort "$ServerPort" -CondaPath "$CondaPath" -InstallMode "$InstallMode" -LogPath "$LogFile" -StatusPath "$INSTDIR\setup_status.txt"'
  ${IfNot} ${FileExists} "$INSTDIR\setup_status.txt"
    MessageBox MB_ICONSTOP "Environment setup failed (no status file). See install.log."
    Abort
  ${EndIf}
  FileOpen $0 "$INSTDIR\setup_status.txt" r
  FileRead $0 $1
  FileClose $0
  ${If} $1 != "OK$\r$\n"
    MessageBox MB_ICONSTOP "Environment setup failed. See install.log."
    Abort
  ${EndIf}

  WriteUninstaller "$INSTDIR\Uninstall.exe"
SectionEnd

Section "Uninstall"
  MessageBox MB_YESNO|MB_ICONQUESTION "Remove the 'multicellpose' Conda environment?" IDYES remove_env
  goto skip_env_removal

remove_env:
  ReadEnvStr $0 COMSPEC
  nsExec::ExecToLog '"$0" /c conda env remove -n multicellpose -y'

skip_env_removal:
  Delete "$APPDATA\Microsoft\Windows\Start Menu\Programs\MultiCellPose.lnk"
  Delete "$INSTDIR\Uninstall.exe"
  RMDir /r "$INSTDIR"
SectionEnd
