libyuv tools

Description:
============

libyuv tool is a package with command line tools prebuilt.
Source is available in the libyuv package.

See https://code.google.com/p/libyuv/

It is released under the same license as the libyuv project.

Files:
======

bin/psnr - tool to compute PSNR, MSE or SSIM for 2 YUV files.

PSNR tool:
==========

Command line help
  psnr [-options] org_seq rec_seq [rec_seq2.. etc]
  options:
   -s <width> <height> .... specify YUV size, mandatory if none of the sequences have the
                            resolution embedded in their filename (ie. bali.1920x800_24Hz_P420.yuv)
   -psnr .................. compute PSNR (default)
   -ssim .................. compute SSIM
   -mse ................... compute MSE
   -swap .................. Swap U and V plane
   -skip <org> <rec> ...... Number of frame to skip of org and rec
   -frames <num> .......... Number of frames to compare
   -t <num> ............... Number of threads
   -yscale ................ Scale org Y values of 16..240 to 0..255
   -n ..................... Show file name
   -v ..................... verbose++
   -q ..................... quiet
   -h ..................... this help

files for org_seq and rec_seq are raw .yuv in I420 planar format.
-psnr -ssim and -mse can all be used at once.

Example:
========

:: Create original YUV and a decoded YUV for comparison
ffmpeg -y -vsync 0 -i _on2HDs.avi -vcodec rawvideo -pix_fmt yuv420p -s 1280x720 -an tulip.1280x720_30Hz_P420.yuv
vpxenc --profile=1 -w 1280 -h 720 --fps=30000/1001 --target-bitrate=1200 tulip.1280x720_30Hz_P420.yuv -o tulip.webm -p 2 --codec=vp8 --good --cpu-used=0 --lag-in-frames=25 --min-q=0 --max-q=63 --end-usage=vbr --auto-alt-ref=1 --kf-max-dist=9999 --kf-min-dist=0 --drop-frame=0 --static-thresh=0 --bias-pct=50 --minsection-pct=0 --maxsection-pct=2000 --arnr-maxframes=7 --arnr-strength=5 --arnr-type=3 --sharpness=0 --undershoot-pct=100 --verbose --psnr -t 4
ffmpeg -y -vsync 0 -i tulip.webm tulip_webm.1280x720_30Hz_P420.yuv

:: Compute PSNR and SSIM
psnr -s 1280 720 tulip_webm.1280x720_30Hz_P420.yuv tulip.1280x720_30Hz_P420.yuv
Frame    PSNR-Y          PSNR-U          PSNR-V          PSNR-All        Frame
Global:  34.960924       40.016864       40.062411       36.094978        500
Avg:     35.080943       40.101847       40.145290       36.204514        500
Min:     32.791396       38.491528       38.522088       34.114365        494

psnr -ssim tulip_webm.1280x720_30Hz_P420.yuv tulip.1280x720_30Hz_P420.yuv
Frame     SSIM-Y          SSIM-U          SSIM-V          SSIM-All       Frame
Avg:      0.879312        0.928988        0.932257        0.896415        500
Min:      0.837683        0.904105        0.909518        0.868070        494

For PSNR, Global PSNR-ALL is preferred: 36.094978
For SSIM, Avg SSim-All: 0.896415

Bugs:
=====

Please report all bugs to our issue tracker:
    http://code.google.com/p/libyuv/issues

