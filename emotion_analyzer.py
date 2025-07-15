def check_consistency(face_em, voice_em):
    if "unknown" in (face_em, voice_em):
        return "Uncertain"
    return "Match" if face_em == voice_em else "Mismatch"