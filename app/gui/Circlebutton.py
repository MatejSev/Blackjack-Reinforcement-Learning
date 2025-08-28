import py

class CircleButton:
    def __init__(self, x, y, radius, text='', color=(0, 0, 255), text_color=(255, 255, 255)):
        self.x = x
        self.y = y
        self.radius = radius
        self.text = text
        self.color = color
        self.text_color = text_color
        self.font = pygame.font.Font(None, 32)
        self.txt_surface = self.font.render(text, True, text_color)
        self.clicked = False
    
    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.radius)
        text_rect = self.txt_surface.get_rect(center=(self.x, self.y))
        screen.blit(self.txt_surface, text_rect)
        
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = event.pos
            distance = ((self.x - mouse_pos[0])**2 + (self.y - mouse_pos[1])**2)**0.5
            if distance <= self.radius:
                self.clicked = True
        
        elif event.type == pygame.MOUSEBUTTONUP:
            if self.clicked:
                mouse_pos = event.pos
                distance = ((self.x - mouse_pos[0])**2 + (self.y - mouse_pos[1])**2)**0.5
                if distance <= self.radius:
                    self.clicked = False
                    return False
                self.clicked = False
        return True
