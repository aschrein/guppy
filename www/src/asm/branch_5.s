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

sample r10.xyzw, t0.xyzw, s0, r0.xy
sample r11.xyzw, t1.xyzw, s0, r0.xy
sample r12.xyzw, t2.xyzw, s0, r0.xy

lerp r10.xyzw, r10.xyzw, r11.xyzw, r12.x

st u0.xyzw, r0.zw, r10.xyzw
ret