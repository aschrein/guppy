jmp ENTRY

; Distance function
; In   : r32.xyz
; Uses : r33.xyzw, r32.xyzw
; Out  : r32.w
DIST_FN:

; Sphere_0
sub.f32 r33.xyz, r32.xyz, f3(0.0 0.0 5.0)
len r33.w, r33.xyz
sub.f32 r33.w, r33.w, f(9.0)

; Sphere_1
sub.f32 r34.xyz, r32.xyz, f3(0.0 0.0 -5.0)
len r34.w, r34.xyz
sub.f32 r34.w, r34.w, f(5.0)

; Smooth min
sub.f32 r34.x, r33.w, r34.w
mul.f32 r34.x, r34.x, f(0.2)
mad.f32 r34.x, r34.x, f(0.5), f(0.5)
clamp r34.x, r34.x
; mul.f32 r34.x, r34.x, r34.x
lerp r32.w, r34.w, r33.w, r34.x

sub.f32 r34.w, r34.x, f(1.0)
mul.f32 r34.w, r34.w, f(5.)
mad.f32 r32.w, r34.x, r34.w, r32.w


pop_mask

ENTRY:
; Figure out where we are in the screen space
mov r0.xy, thread_id
and r0.x, r0.x, u(255)
div.u32 r0.y, r0.y, u(256)
mov r0.zw, r0.xy

; put the red color as an indiacation of ongoing work
st u0.xyzw, r0.zw, f4(1.0 0.0 0.0 1.0)

; Normalize screen coordiantes
utof r0.xy, r0.xy
; add 0.5 to fit the center of the texel
add.f32 r0.xy, r0.xy, f2(0.5 0.5)
; normalize coordinates
div.f32 r0.xy, r0.xy, f2(256.0 256.0)
; tx * 2.0 - 1.0
mul.f32 r0.xy, r0.xy, f2(2.0 -2.0)
sub.f32 r0.xy, r0.xy, f2(1.0 -1.0)

; Setup a simple pinhole camera
; Camera position
mov r1.xyz, f3(10.0 10.0 0.0)
; Camera look vector
mov r2.xyz, f3(-0.7071 -0.7071 0.0)
; Camera right vector
mov r3.xyz, f3(-0.7071 0.7071 0.0)
; Camera up vector
mov r4.xyz, f3(0.0 0.0 1.0)
; Setup ray direction
mov r5.xyz, r2.xyz
mad.f32 r5.xyz, r0.xxx, r3.xyz, r5.xyz
mad.f32 r5.xyz, r0.yyy, r4.xyz, r5.xyz
norm r5.xyz, r5.xyz

; Now solve the scene

mov r15.xyz, r5.xyz
mul.f32 r15.xyz, r15.xyz, f3(0.01 0.01 0.01)
add.f32 r15.xyz, r15.xyz, r1.xyz

;jmp LOOP_END

push_mask LOOP_END
LOOP_BEGIN:
; if (r16.y < 16)
lt.u32 r16.x, r16.y, u(16)
mask_nz r16.x
; Loop body begin
mov r32.xyz, r15.xyz
push_mask RET
jmp DIST_FN
RET:
gt.f32 r14.x, r32.w, f(0.001)
sub.u32 r13.x, u(1), r14.x
utof r13.x, r13.x
mask_nz r14.x
mad.f32 r15.xyz, r5.xyz, r32.www, r15.xyz

; Loop body end
; Increment iteration counter
add.u32 r16.y, r16.y, u(1)

jmp LOOP_BEGIN

LOOP_END:

mov r10.w, f(1.0)
abs.f32 r10.xyz, r5.www

push_mask L1
mask_nz r13.x
norm r15.xyz, r15.xyz
sample r10.xyzw, t0.xyzw, s0, r15.xy
pop_mask
L1:
; mov r5.xyz, r32.www
; mov r16.y, u(1)
; utof r14.x, r16.y
; div.f32 r14.x, r14.x, f(4.0)

st u0.xyzw, r0.zw, r10.xyzw
ret