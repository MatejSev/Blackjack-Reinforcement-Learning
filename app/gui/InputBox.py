import pygame

class InputBox:
    def __init__(self, x, y, w, h, active_color=(0,0,255), inactive_color=(0,0,0)):
        self.rect = pygame.Rect(x, y, w, h)
        self.inactive_color = inactive_color
        self.active_color = active_color
        self.color = self.inactive_color
        self.text = ''
        self.txt_surface = pygame.font.Font(None, 32).render(self.text, True, self.color)
        self.active = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.active = not self.active
            else:
                self.active = False
            self.color = self.active_color if self.active else self.inactive_color
        if event.type == pygame.KEYDOWN:
            if self.active:
                if event.key == pygame.K_RETURN:
                    return False
                elif event.key == pygame.K_BACKSPACE:
                    self.text = self.text[:-1]
                else:
                    self.text += event.unicode
                self.txt_surface = pygame.font.Font(None, 32).render(self.text, True, self.color)
        return True
    
    def draw(self, screen):
        screen.blit(self.txt_surface, (self.rect.x+5, self.rect.y+5))
        pygame.draw.rect(screen, self.color, self.rect, 2)

    def get_text(self):
        text = self.text
        self.text = ''
        self.txt_surface = pygame.font.Font(None, 32).render(self.text, True, self.color)
        return text
