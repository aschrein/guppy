; uint lane_id = get_lane_id();
    mov r0.x, lane_id
; if (lane_id & 1) {
    push_mask BRANCH_END
    and r0.y, r0.x, u(1)
    mask_nz r0.y
LOOP_BEGIN:
    ; // Do smth
     mov r0.x, r0.x
     mov r0.x, r0.x
     mov r0.x, r0.x
     mov r0.x, r0.x
     mov r0.x, r0.x
     
    pop_mask                ; pop mask and reconverge
BRANCH_END:
    ; // Do some more
    ret