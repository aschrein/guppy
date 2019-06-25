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

sample r12.xyzw, t4.xyzw, s0, r0.xy

mov r32.w, f(1.0)

push_mask SKIP_S1
gt.f32 r15.x, r12.x, f(0.0)
mask_nz r15.x
mov r32.y, f(1.0)
st u0.xyzw, r0.zw, r32.xyzw
sample r10.xyzw, t0.xyzw, s0, r0.xy
pop_mask

SKIP_S1:

push_mask SKIP_S2
lt.f32 r16.x, r12.x, f(1.0)
mask_nz r16.x
mov r32.z, f(1.0)
st u0.xyzw, r0.zw, r32.xyzw
sample r11.xyzw, t1.xyzw, s0, r0.xy
pop_mask

SKIP_S2:


lerp r10.xyzw, r10.xyzw, r11.xyzw, r12.xxxx

mov r10.x, clock
utof r10.x, r10.x
div.f32 r10.x, r10.x, f(120.0)
mov r10.yz, f2(0.0 0.0)
st u0.xyzw, r0.zw, r10.xxxw
ret