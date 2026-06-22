"""Compile .po files to .mo files without GNU gettext."""
import os
import struct

def compile_po(po_path, mo_path):
    """Minimal .po -> .mo compiler with proper metadata."""
    messages = {}
    msgid = None
    msgstr_lines = []
    in_msgstr = False
    in_msgid = False
    current_msgid_lines = []
    
    with open(po_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Skip comments
            if line.startswith('#'):
                continue
            
            if line.startswith('msgid_plural ') or line.startswith('msgstr['):
                # Skip plural forms for now (basic compiler)
                in_msgstr = False
                in_msgid = False
                continue
            
            if line.startswith('msgid '):
                # Save previous pair
                if msgid is not None:
                    msgstr_val = ''.join(msgstr_lines)
                    if msgstr_val:
                        messages[msgid] = msgstr_val
                
                in_msgid = True
                in_msgstr = False
                txt = line[6:].strip('"')
                current_msgid_lines = [txt.replace('\\n', '\n').replace('\\"', '"')]
                msgstr_lines = []
                
            elif line.startswith('msgstr '):
                msgid = ''.join(current_msgid_lines)
                in_msgstr = True
                in_msgid = False
                txt = line[7:].strip('"')
                msgstr_lines = [txt.replace('\\n', '\n').replace('\\"', '"')]
                
            elif line.startswith('"') and line.endswith('"'):
                val = line[1:-1].replace('\\n', '\n').replace('\\"', '"')
                if in_msgid:
                    current_msgid_lines.append(val)
                elif in_msgstr:
                    msgstr_lines.append(val)
                    
            elif line == '':
                # Empty line - save current pair
                if msgid is not None and in_msgstr:
                    msgstr_val = ''.join(msgstr_lines)
                    if msgstr_val:
                        messages[msgid] = msgstr_val
                    msgid = None
                    in_msgstr = False
                    in_msgid = False
    
    # Last entry
    if msgid is not None and in_msgstr:
        msgstr_val = ''.join(msgstr_lines)
        if msgstr_val:
            messages[msgid] = msgstr_val
    
    # Build .mo file
    keys = sorted(messages.keys())
    
    offsets = []
    ids = b''
    strs = b''
    
    for key in keys:
        key_bytes = key.encode('utf-8')
        val_bytes = messages[key].encode('utf-8')
        offsets.append((len(ids), len(key_bytes), len(strs), len(val_bytes)))
        ids += key_bytes + b'\0'
        strs += val_bytes + b'\0'
    
    n = len(keys)
    keystart = 28 + n * 8 + n * 8
    valuestart = keystart + len(ids)
    
    output = bytearray()
    # Magic number
    output += struct.pack('I', 0x950412de)
    # Version
    output += struct.pack('I', 0)
    # Number of strings
    output += struct.pack('I', n)
    # Offset of table with original strings
    output += struct.pack('I', 28)
    # Offset of table with translation strings
    output += struct.pack('I', 28 + n * 8)
    # Size of hashing table
    output += struct.pack('I', 0)
    # Offset of hashing table
    output += struct.pack('I', 0)
    
    # Original strings table
    for o in offsets:
        output += struct.pack('II', o[1], keystart + o[0])
    
    # Translated strings table
    for o in offsets:
        output += struct.pack('II', o[3], valuestart + o[2])
    
    # String data
    output += ids
    output += strs
    
    os.makedirs(os.path.dirname(mo_path), exist_ok=True)
    with open(mo_path, 'wb') as f:
        f.write(output)
    
    return n

if __name__ == '__main__':
    base = os.path.dirname(os.path.abspath(__file__))
    locale_dir = os.path.join(base, 'locale')
    
    for lang in ['fr', 'ar']:
        po = os.path.join(locale_dir, lang, 'LC_MESSAGES', 'django.po')
        mo = os.path.join(locale_dir, lang, 'LC_MESSAGES', 'django.mo')
        if os.path.exists(po):
            count = compile_po(po, mo)
            print(f"  Compiled: {lang}/LC_MESSAGES/django.mo ({count} strings)")
        else:
            print(f"  Missing: {po}")
    
    print("Done!")
