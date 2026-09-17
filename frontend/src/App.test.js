import { render, screen } from '@testing-library/react';
import axios from 'axios';
import App from './App';

jest.mock('axios');

test('renders and loads the AI Crime Scene Investigation dashboard', async () => {
  axios.get.mockResolvedValueOnce({ data: [] });

  render(<App />);

  expect(
    await screen.findByText(/No cases yet/i)
  ).toBeInTheDocument();

  expect(
    screen.getByRole('heading', {
      name: /AI Crime Scene Investigation/i
    })
  ).toBeInTheDocument();

  expect(
    screen.getByRole('heading', {
      name: /Create New Case/i
    })
  ).toBeInTheDocument();
});