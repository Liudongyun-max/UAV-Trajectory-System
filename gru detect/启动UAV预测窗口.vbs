Set shell = CreateObject("WScript.Shell")
scriptDir = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
cmd = "cmd /c set KMP_DUPLICATE_LIB_OK=TRUE && set OMP_NUM_THREADS=1 && cd /d """ & scriptDir & """ && conda run -n yolo pythonw """ & scriptDir & "\predict_gui.py"""
shell.Run cmd, 0, False
