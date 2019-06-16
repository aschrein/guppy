    mov r4.w, lane_id
    utof r4.xyzw, r4.wwww
    mov r4.z, wave_id
    utof r4.z, r4.z
    add.f32 r4.xyzw, r4.xyzw, f4(0.0 0.0 0.0 1.0)
    lt.f32 r4.xy, r4.ww, f2(16.0 8.0)
    utof r4.xy, r4.xy
    br_push r4.x, LB_1, LB_2
    mov r0.x, f(666.0)
    br_push r4.y, LB_0_1, LB_0_2
    mov r0.y, f(666.0)
    pop_mask
LB_0_1:
    mov r0.y, f(777.0)
    pop_mask
LB_0_2:
    pop_mask
LB_1:
    mov r0.x, f(777.0)

    ; push the current wave mask
    push_mask LOOP_END
LOOP_PROLOG:
    lt.f32 r4.x, r4.w, f(32.0)
    add.f32 r4.w, r4.w, f(1.0)
    ; Setting current lane mask
    ; If all lanes are disabled pop_mask is invoked
    ; If mask stack is empty then wave is retired
    mask_nz r4.x
LOOP_BEGIN:
    jmp LOOP_PROLOG
LOOP_END:
    pop_mask
    
    
LB_2:
    mov r4.y, lane_id
    utof r4.y, r4.y
    ret